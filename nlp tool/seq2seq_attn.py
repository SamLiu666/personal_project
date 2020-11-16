# [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import utils
import tensorflow_addons as tfa
import pickle
from visual import seq2seq_attention
"""Reference by https://github.com/MorvanZhou/NLP-Tutorials
Data: input_len: 8, output_len:11
"""

# 设置gpu内存自增长
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class seq2seqAttn(keras.Model):
    def __init__(self, enc_v_dim, dec_v_dim, emb_dim, units,
                 attention_layer_size, max_pred_len, start_token, end_token):
        super(seq2seqAttn, self).__init__()
        super().__init__()
        self.units = units

        # prediction restriction
        self.max_pred_len = max_pred_len
        self.start_token = start_token
        self.end_token = end_token

        # encoder
        self.enc_embeddings = keras.layers.Embedding(
            input_dim=enc_v_dim, output_dim=emb_dim,    # [enc_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        self.encoder = keras.layers.LSTM(units=units, return_sequences=True, return_state=True)

        # decoder with attention
        self.attention = tfa.seq2seq.LuongAttention(units, memory=None, memory_sequence_length=None)
        self.decoder_cell = tfa.seq2seq.AttentionWrapper(
            cell=keras.layers.LSTMCell(units=units),
            attention_mechanism=self.attention,
            attention_layer_size=attention_layer_size,
            alignment_history=True,                     # for attention visualization
        )

        self.dec_embeddings = keras.layers.Embedding(
            input_dim=dec_v_dim, output_dim=emb_dim,    # [dec_n_vocab, emb_dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.1),
        )
        decoder_dense = keras.layers.Dense(dec_v_dim)   # output layer

        # train decoder
        self.decoder_train = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.TrainingSampler(),   # sampler for train
            output_layer=decoder_dense
        )
        self.cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.05, clipnorm=5.0)

        # predict decoder
        self.decoder_eval = tfa.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            sampler=tfa.seq2seq.sampler.GreedyEmbeddingSampler(),       # sampler for predict
            output_layer=decoder_dense
        )


    def encode(self, x):
        """:arg  x: [batch, seq_len]
        return:
        out_put: [batch, seq_len, units]
        hidden_state: [ seq_len, units]
        cell_state: [ seq_len, units]
        """
        o = self.enc_embeddings(x) # [ batch, seq_len, emb_dim)
        init_s = [tf.zeros((x.shape[0], self.units)), tf.zeros((x.shape[0], self.units))]
        o, h, c = self.encoder(o, initial_state=init_s)
        return o, h, c

    def set_attention(self, x):
        """
        :arg  加工encode 的信息
        x: [batch, seq_len]
        out_put: [batch, seq_len, units]
        hidden_state: [ seq_len, units]
        cell_state: [ seq_len, units]

        """
        o, h, c = self.encode(x)
        # encoder output for attention to focus
        self.attention.setup_memory(o)
        # wrap state by attention wrapper
        s = self.decoder_cell.get_initial_state(batch_size=x.shape[0], dtype=tf.float32).clone(cell_state=[h, c])
        #print("set_attention: ", s)
        return s

    def train_logits(self, x, y, seq_len):
        """:arg
        1.拿到encoder的attention信息和state；
        2.筛选出标签；
        3.把标签在decoder中embed；
        4.拿着所有缓存的 encoded state (attention memory) 和 encoder最后一步产生 state，放入decoder预测，得到所有的由加了注意力的output。
        """
        s = self.set_attention(x)
        dec_in = y[:, :-1]   # ignore <EOS>
        dec_emb_in = self.dec_embeddings(dec_in)
        o, _, _ = self.decoder_train(dec_emb_in, s, sequence_length=seq_len)
        logits = o.rnn_output
        #print(" train_logits: ", logits)
        return logits

    def step(self, x, y, seq_len):
        """每一步训练， 将正确标签作为decoder输入"""
        with tf.GradientTape() as tape:
            logits = self.train_logits(x, y, seq_len)
            dec_out = y[:, 1:]  # ignore <GO>
            loss = self.cross_entropy(dec_out, logits)
            grads = tape.gradient(loss, self.trainable_variables)
            #print("step: ", grads)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()

    def inference(self, x, return_align=False):
        """预测"""
        s = self.set_attention(x)   # 获取encoder的attention 信息
        done, i, s = self.decoder_eval.initialize(
            self.dec_embeddings.variables[0],
            start_tokens=tf.fill([x.shape[0], ], self.start_token),
            end_token=self.end_token,
            initial_state=s,
        )

        pred_id = np.zeros((x.shape[0], self.max_pred_len), dtype=np.int32)
        for l in range(self.max_pred_len):
            o, s, i, done = self.decoder_eval.step(
                time=l, inputs=i, state=s, training=False)
            pred_id[:, l] = o.sample_id
            #print("inference: ", pred_id)
        if return_align:
            return np.transpose(s.alignment_history.stack().numpy(), (1, 0, 2))
        else:
            s.alignment_history.mark_used()  # otherwise gives warning
            return pred_id


def train():
    data = utils.DateData(2000)
    print("Chinese time order: yy/mm/dd ", data.date_cn[:3], "\nEnglish time order: dd/M/yyyy ", data.date_en[:3])
    print("vocabularies: ", data.vocab)
    print("x index sample: \n{}\n{}".format(data.idx2str(data.x[0]), data.x[0]),
          "\ny index sample: \n{}\n{}".format(data.idx2str(data.y[0]), data.y[0]))
    print("data.x[0]), data.x[0] length: ", len(data.x[0]), len(data.y[0]))
    print("data.num_word: ", data.num_word)

    model = seq2seqAttn(
        data.num_word, data.num_word, emb_dim=12, units=14, attention_layer_size=16,
        max_pred_len=11, start_token=data.start_token, end_token=data.end_token)

    # training
    for t in range(1001):
        bx, by, decoder_len = data.sample(64)
        loss = model.step(bx, by, decoder_len)
        if t % 100 == 0:
            target = data.idx2str(by[0, 1:-1])
            pred = model.inference(bx[0:1])
            res = data.idx2str(pred[0])
            src = data.idx2str(bx[0])
            print(
                "t: ", t,
                "| loss: %.5f" % loss,
                "| input: ", src,
                "| target: ", target,
                "| inference: ", res,
            )

    pkl_data = {"i2v": data.i2v, "x": data.x[:6], "y": data.y[:6], "align": model.inference(data.x[:6], return_align=True)}

    with open("./visual/tmp/attention_align.pkl", "wb") as f:
        pickle.dump(pkl_data, f)


if __name__ == "__main__":
    import time
    start = time.time()
    train()
    end = time.time()
    print("seq2seq_attention: ",end - start)

    seq2seq_attention()