import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from .layers.Embed import PatchEmbedding
from .layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer


from math import ceil

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2
        self.task_name = configs.task_name

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if l == 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
                            1, configs.dropout,
                            self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l), configs.factor
                            ) for l in range(configs.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model))

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(configs, (self.pad_out_len // self.seg_len), configs.factor, configs.d_model, configs.n_heads,
                                           configs.d_ff, configs.dropout),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    self.seg_len,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    # activation=configs.activation,
                )
                for l in range(configs.e_layers + 1)
            ],
        )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d = n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)

        return dec_out

    def anomaly_detection(self, x_enc):
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_out = self.head(enc_out[-1].permute(0, 1, 3, 2)).permute(0, 2, 1)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
    
import os
import yaml
from models.model_base import MultiVarModelBase 
import argparse
    
class CrossFormerArgs:
    def __init__(self):
        pass
    def add_argument(self, arg:str, type, default, help='none' ,required=False, nargs='', action=''):
        if arg.startswith('--'):
            args_name = arg[2:]
            try:
                setattr(self, args_name, type(default))
            except:
                setattr(self, args_name, default)

class CrossFormer_MultiVarModel(MultiVarModelBase):
    def __init__(self, configs: dict = ...) -> None:
        super().__init__(configs)
        self.configs = configs
        self.init_model()
    def init_model(self, args=...) -> nn.Module:
        # task configs
        self.tensor_shape = self.configs['tensor_shape']
        self.input_len = self.configs['his_len']
        self.pred_len = self.configs['pred_len']
        self.normalizer = self.configs['normalizer']
        # model parameters config
        model_configs_yaml = os.path.join( os.path.dirname(__file__), 'model.yml' )
        model_configs = yaml.safe_load(open(model_configs_yaml))
        
        self.e_layers = model_configs['e_layers']
        self.d_layers = model_configs['d_layers']
        self.factor = model_configs['factor']
        self.enc_in = self.tensor_shape[0]
        self.c_out = self.tensor_shape[0]
        self.dec_in = self.tensor_shape[0]
        self.d_model = model_configs['d_model']
        self.d_ff = model_configs['d_ff']
        self.loss = model_configs['loss']
        self.n_heads = model_configs['n_heads']


        # parser = argparse.ArgumentParser(description='TimesNet')
        parser = CrossFormerArgs()

        # basic config
        parser.add_argument('--task_name', type=str, default='long_term_forecast',
                            help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        # parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
        parser.add_argument('--model_name', type=str, default='CrossFormer', help='model id')

        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=self.input_len, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=self.input_len, help='start token length')
        parser.add_argument('--pred_len', type=int, default=self.pred_len, help='prediction sequence length')
        parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

        # inputation task
        parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

        # anomaly detection task
        parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

        # model define
        parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
        parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        parser.add_argument('--enc_in', type=int, default=self.enc_in, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=self.dec_in, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=self.c_out, help='output size')
        parser.add_argument('--d_model', type=int, default=self.d_model, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=self.e_layers, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=self.d_layers, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=self.factor, help='attn factor')
        parser.add_argument('--distil', action='store_false', type=bool,
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', type = bool, default= False, help='whether to output attention in ecoder')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', type = bool, default= False, help='use automatic mixed precision training')

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', type = bool, default= False, help='use multiple gpus')
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        # de-stationary projector params
        parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                            help='hidden layer dimensions of projector (List)')
        parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

        
        # self.configs = parser.parse_args()
        # self.configs, unknown = parser.parse_known_args()
        self.configs = parser
        self.configs.d_ff = self.d_ff
        self.configs.n_heads = self.n_heads
        # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        # if args.use_gpu and args.use_multi_gpu:
        #     args.devices = args.devices.replace(' ', '')
        #     device_ids = args.devices.split(',')
        #     args.device_ids = [int(id_) for id_ in device_ids]
        #     args.gpu = args.device_ids[0]

        self.model = Model(self.configs).float()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)
        if self.loss == 'mse':
            self.criterion = nn.MSELoss()
        elif self.loss == 'mae':
            self.criterion = nn.L1Loss()
        args = self.configs
        setting = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.model_name,
                args.task_name,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des)
        print(setting)
        torch.cuda.empty_cache()

###################
    def forward(self, x, aux_info: dict = ...):
            # x [batch, time, dim1, dim2]
            # TimesNet [batch, time, dim1*dim2]
            # print(x.shape);exit()
            batch, time, dim1, dim2 = x.size()
            value = x.view(batch, time, dim1*dim2)
            in_data = value[:, :self.input_len, :]
            truth = value[:, self.input_len:self.input_len+self.pred_len, :]

            dec_inp = torch.zeros_like(truth).float().to(x.device)
            dec_inp = torch.cat([in_data, dec_inp], dim=1).float().to(x.device)
            # normalization
            in_data = self.normalizer.transform(in_data)
            pred = self.model(in_data, None, dec_inp, None)
            # inverse
            pred = self.normalizer.inverse_transform(pred)

            return pred, truth
            
    def backward(self, loss):
        self.model.zero_grad()
        loss.backward()
        self.optim.step()
    
    def get_loss(self, pred, truth):
        loss = self.criterion(pred, truth)
        return loss