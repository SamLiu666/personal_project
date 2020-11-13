from tensorflow import keras
import tensorflow as tf
from utils import process_w2v_data
from visual import show_w2v_word_embedding
"""Reference by https://github.com/MorvanZhou/NLP-Tutorials"""


class CBOW(keras.Model):
    def __init__(self, vocab_d, emb_dim):
        """
        vocab_d: 词汇表数目, int
        emb_dim: 单词维度， int
        输出：[vocab_d, emb_dim]
        """
        super().__init__()
        self.vocab_d = vocab_d
        self.emb_dim = emb_dim
        """  Input shape:2D tensor with shape: `(batch_size, input_length)`.
             Output shape:3D tensor with shape: `(batch_size, input_length, output_dim)`."""
        self.embeddings = keras.layers.Embedding(
            input_dim=vocab_d, output_dim=emb_dim,
            embeddings_initializer='uniform' )
            # embeddings_initializer=keras.initializers.RandomNormal(0., 0.1),

        # 负采样(negative sampling:)参数
        self.nce_w = self.add_weight(
            name="nce_w", shape=[vocab_d, emb_dim],
            initializer=keras.initializers.TruncatedNormal(0., 0.1))  # [n_vocab, emb_dim]
        self.nce_b = self.add_weight(
            name="nce_b", shape=(vocab_d,),
            initializer=keras.initializers.Constant(0.1))  # [n_vocab, ]

        self.opt = keras.optimizers.Adam(0.01)

    def call(self, x, training=None, mask=None):
        # x: input [batch, window_size*2]
        output = self.embeddings(x)   # [batch, window_size*2, emb_dim]
        output = tf.reduce_mean(output, axis=1)  # [batch, emb_dim] 窗口值进行求和平均值
        return output

    def loss(self, x, y, training=None):
        """:arg
        x: input [batch, window_size*2]
        y: output [batch, ]
        """
        embedding_output = self.call(x, training)   # [batch, emb_dim]
        nce_loss = tf.nn.nce_loss(
            weights=self.nce_w, biases=self.nce_b, labels=tf.expand_dims(y, axis=1),
            inputs=embedding_output, num_sampled=5, num_classes=self.vocab_d
        )
        #print("nce_loss: ", nce_loss)
        # If axis is None, all dimensions are reduced, and a tensor with a single element is returned.
        return tf.reduce_mean(nce_loss)

    def step(self, x, y):
        # 每一步更新梯度
        with tf.GradientTape() as tape:
            loss = self.loss(x, y, True)
            grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients((zip(grads, self.trainable_variables)))
        return loss.numpy()


class SkipGram(CBOW):
    def __init__(self,vocab_d, emb_dim):
        """继承CBOW，更改call_method：Embedding 之后不需要再进行sum and reduce"""
        super().__init__(vocab_d, emb_dim)

    def call(self, x, training=None, mask=None):
        """x: input [batch, ]
        return :  [batch, emb_dim]"""
        return self.embeddings(x)   # [batch, emb_dim]


def train(model, data):
    for t in range(2000):
        bx, by = data.sample(64)  # batch=8
        loss = model.step(bx, by)
        if t%200 == 0:
            print("step: {} | loss: {}".format(t, loss))


class SkipGram_2(keras.Model):
    def __init__(self, vocab_d, emb_dim):
        super().__init__()
        self.vocab_d = vocab_d
        self.emb_dim = emb_dim
        self.embeddings = keras.layers.Embedding(
            input_dim=vocab_d, output_dim=emb_dim,
            embeddings_initializer='uniform')
        # noise-contrastive estimation
        self.nce_w = self.add_weight(
            name="nce_w", shape=[vocab_d, emb_dim],
            initializer=keras.initializers.TruncatedNormal(0., 0.1))  # [n_vocab, emb_dim]
        self.nce_b = self.add_weight(
            name="nce_b", shape=(vocab_d,),
            initializer=keras.initializers.Constant(0.1))  # [n_vocab, ]

        self.opt = keras.optimizers.Adam(0.01)

    def call(self, x, training=None, mask=None):
        # x: input [batch, ]
        return self.embeddings(x)   # [batch, emb_dim]

    def loss(self, x, y, training=None):
        embedded = self.call(x, training)
        return tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.nce_w, biases=self.nce_b, labels=tf.expand_dims(y, axis=1),
                inputs=embedded, num_sampled=5, num_classes=self.vocab_d))

    def step(self, x, y):
        with tf.GradientTape() as tape:
            loss = self.loss(x, y, True)
            grads = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()


if __name__ == '__main__':
    import time
    cost_time = {}
    corpus = [
        # numbers
        "5 2 4 8 6 2 3 6 4",
        "4 8 5 6 9 5 5 6",
        "1 1 5 2 3 3 8",
        "3 6 9 6 8 7 4 6 3",
        "8 9 9 6 1 4 3 4",
        "1 0 2 0 2 1 3 3 3 3 3",
        "9 3 3 0 1 4 7 8",
        "9 9 8 5 6 7 1 2 3 0 1 0",

        # alphabets, expecting that 9 is close to letters
        "a t g q e h 9 u f",
        "e q y u o i p s",
        "q o 9 p l k j o k k o p",
        "h g y i u t t a e q",
        "i k d q r e 9 e a d",
        "o p d g 9 s a f g a",
        "i u y g h k l a s w",
        "o l u y a o g f s",
        "o p i u y g d a s j d l",
        "u k i l o 9 l j s",
        "y g i s h k j l f r f",
        "i o h n 9 9 d 9 f a 9",
    ]
    ##########################################
    start = time.time()
    d = process_w2v_data(corpus, skip_window=2, method="cbow")
    model = CBOW(vocab_d=d.num_word, emb_dim=4)
    train(model, d)
    print("CBOW 训练结果： ",d.num_word, model.embeddings.variables)
    end = time.time()
    cost_time["CBOW"] = end - start
    print("训练时间： ", end - start)
    # plotting
    show_w2v_word_embedding(model, d, "./visual/results/cbow.png")

    ##########################################
    start = time.time()
    d = process_w2v_data(corpus, skip_window=2, method="skip_gram")
    model_2 = SkipGram(vocab_d=d.num_word, emb_dim=4)
    train(model_2, d)
    print("CBOW 训练结果： ", d.num_word, model_2.embeddings.variables)
    end = time.time()
    cost_time["CBOW"] = end - start
    print("训练时间： ", end - start)

    # plotting
    show_w2v_word_embedding(model_2, d, "./visual/results/skip.png")
    print(cost_time)