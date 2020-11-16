import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils
import tensorflow_addons as tfa


class Seq2Seq(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units, max_pred_len, start_token, end_token):
        super().__init__()
        self.units = units
        self.max_pred_len = max_pred_len  # 序列最大长度
        self.start_token= start_token
        self.end_token = end_token

        # encoder
        self.enc_embeddings = keras.layers.Embedding(
            input_dim=enc_v_dim, output_dim=emb_dim,  # [enc_n_vocab, emb_dim]
            embeddings_initializer='uniform',
            # embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )

        # encoder就可以将原始的词向量按顺序组装起来变成句向量,
        # return_sequences=True, return_state=True
        # stm1 存放的就是全部时间步的 hidden state。
        # state_h 存放的是最后一个时间步的 hidden state
        # state_c 存放的是最后一个时间步的 cell state"
        self.encoder = keras.layers.LSTM(units=units,
                                         return_sequences=True,
                                         return_state=True)

        # decoder
        self.dec_embeddings = keras.layers.Embedding(
            input_dim=dec_v_dim, output_dim=emb_dim,  # [dec_v_dim, emb_dim]
            embeddings_initializer='uniform',
            # embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        self.decoder_cell = keras.layers.LSTMCell(units=units)  # This class processes one step within the whole time sequence input, whereas tf.keras.layer.LSTM processes the whole sequence
        decoder_dense = keras.layers.Dense(dec_v_dim)  # 输出词的softmax: [output_vocab_len, ]

        # decoder for training
        self.decoder_train = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.TrainingSampler(),
            output_layer=decoder_dense
        )

        # decoder for predicting
        self.decoder_eval = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(),
            output_layer=decoder_dense
        )

        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.001)

    def encode(self, x):
        # x: [batch, seq_len]
        embedded = self.enc_embeddings(x)  # [batch, seq_len, emb_dim]
        init_s = [tf.zeros((x.shape[0], self.units)), tf.zeros((x.shape[0], self.units))]
        out_put, hidden_state, cell_state = self.encoder(embedded, initial_state=init_s)
        # out_put: [batch, seq_len, units]
        # hidden_state: [ seq_len, units]
        # cell_state: [ seq_len, units]
        return [hidden_state, cell_state]

    def train_logits(self, x, y, seq_len):
        """
        训练时，输入的标签为正确的标签，加快训练速度，更快收敛
        x: [batch, input_seq_len]
        y:  [batch, output_seq_len]
        """
        s = self.encode(x)  #  [hidden_state, cell_state]  shape is [seq_len, units]
        dec_in = y[:, :-1]  # 去掉 <EOS>, [batch, output_seq_len-1]

        dec_embd_in = self.dec_embeddings(dec_in)  # [batch, output_seq_len-1, units]
        o, _, _, = self.decoder_train(dec_embd_in, s, sequence_length=seq_len)
        logits = o.rnn_output  # (batch, output_seq_len-1, dec_v_dim) dec_v_dim：输出词汇长度
        return logits

    def step(self, x, y, seq_len):
        with tf.GradientTape() as tape:
            logits = self.train_logits(x, y, seq_len)  # 训练结果：(batch, output_seq_len-1, dec_v_dim)
            dec_out = y[:, 1:]   # 忽略 <GO>  （batch, output_seq_len-1）

            loss = self.cross_entropy(dec_out, logits)  # softmax ：交叉熵
            grads = tape.gradient(loss, self.trainable_variables)  # 更新梯度
        self.opt.apply_gradients(zip(grads, self.trainable_variables))  # 打包统一更新
        return loss.numpy()  # 记录输出，返回

    #def predict_output(self, x):
    def inference(self, x):
        s = self.encode(x)  #x: [batch, seq_len]
        # 解码器初始状态： (1,embedding_size) (1,units), (1,units)
        _, i, s = self.decoder_eval.initialize(
            self.dec_embeddings.variables[0],
            start_tokens=tf.fill([x.shape[0], ], self.start_token),
            end_token=self.end_token,
            initial_state=s,
        )
        #print("inference: ", 1, i, s)
        pred_id = np.zeros((x.shape[0], self.max_pred_len), dtype=np.int32)
        for l in range(self.max_pred_len):
            o, s, i, done = self.decoder_eval.step(
                time=l, inputs=i, state=s, training=False)
            # o = self.decoder_eval.step(
            #     time=l, inputs=i, state=s, training=False)
            #print("inference: ",o, type(o))
            pred_id[:, l] = o.sample_id
            #print("pred_id: ")
        return pred_id

    def predict_output(self, x):
        s = self.encode(x)
        # done, i, s = self.decoder_eval.initialize(
        #     self.dec_embeddings.variables[0],
        #     start_tokens=tf.fill([x.shape[0], ], self.start_token),
        #     end_token=self.end_token,
        #     initial_state=s,
        # )
        _, i, s = self.decoder_eval.initialize(
            self.dec_embeddings.variables[0],
            start_tokens=tf.fill([x.shape[0], ], self.start_token),
            end_token=self.end_token,
            initial_state=s,
        )
        #print("inference: ", 1, i, s)
        pred_id = np.zeros((x.shape[0], self.max_pred_len), dtype=np.int32)
        for l in range(self.max_pred_len):
            o, s, i, done = self.decoder_eval.step(
                time=l, inputs=i, state=s, training=False)
            # o = self.decoder_eval.step(
            #     time=l, inputs=i, state=s, training=False)
            #print("inference: ",o, type(o))
            pred_id[:, l] = o.sample_id
            #print("pred_id: ")
        return pred_id


def train():
    # 设置gpu内存自增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # get and process data
    data = utils.DateData(4000)
    print("Chinese time order: yy/mm/dd ", data.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", data.date_en[:3])
    print("vocabularies: ", data.vocab)
    print("x index sample: \n{}\n{}".format(data.idx2str(data.x[0]), data.x[0]),
          "\ny index sample: \n{}\n{}".format(data.idx2str(data.y[0]), data.y[0]))

    model = Seq2Seq(
        data.num_word, data.num_word, emb_dim=16, units=32,
        max_pred_len=11, start_token=data.start_token, end_token=data.end_token)

    # training
    for t in range(1500):
        bx, by, decoder_len = data.sample(4)
        loss = model.step(bx, by, decoder_len)

        if t % 70 == 0:
            target = data.idx2str(by[0, 1:-1])
            #pred = model.inference(bx[0:1])
            pred = model.predict_output(bx[0:1])
            res = data.idx2str(pred[0])
            src = data.idx2str(bx[0])
            print(
                "t: ", t,
                "| loss: %.3f" % loss,
                "| input: ", src,
                "| target: ", target,
                "| inference: ", res,
            )


if __name__ == '__main__':
    train()