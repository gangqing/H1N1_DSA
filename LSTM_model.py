from tensorflow.compat import v1 as tf
import numpy as np
from H1N1_DSA.GeneData_2 import Data


class GeneConfig:
    def __init__(self):
        tf.disable_eager_execution()
        self.hongkong_path_2009 = "data/GeneFastaResults_HongKong_2009.fasta"
        self.save_path = "models/{name}".format(name=self.get_name())
        self.logdir = "logs".format(name=self.get_name())
        self.num_chars = 4
        self.ds = None
        self.batch_size = 10
        self.new_model = True
        self.epoches = 100
        self.lr = 0.00001
        self.num_units = 10
        # todo
        self.gene_length = 1701  # 一个样本的基因长度
        self.training = True
        self.read_ds()

    def get_name(self):
        return "gene"

    def read_ds(self):
        if self.ds is None:
            self.ds = Data(self)


class SubTensors:
    def __init__(self, config: GeneConfig):
        self.config = config
        # 训练网络
        self.x = tf.placeholder(tf.int64, [None, config.gene_length], name="x")  # [-1, 1701]
        self.lr = tf.placeholder(dtype=tf.float32, shape=None, name="lr")
        self.inputs = [self.x]
        x = tf.one_hot(self.x, config.num_chars, dtype=tf.float32)  # [-1, 1701, 4]

        vec = self.encoder_frame(x, "encoder_frame")  # [-1, 128]

        logits = self.decoder_frame(vec, "decoder_frame")  # [-1, 1701, 4]
        self.predict_y = tf.argmax(logits, axis=2)  # [-1, 1701]

        self.loss(x, logits)

    def loss(self, x, logits):
        """
        :param x: [-1, 1701, 4]
        :param logits: [-1, 1701, 4]
        :return:
        """""
        softmax = tf.nn.softmax_cross_entropy_with_logits_v2(x, logits, axis=2)  # [-1, 1701]
        loss = tf.reduce_mean(softmax)
        opt = tf.train.AdamOptimizer(self.lr)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):  # 控制依赖
            self.train_op = opt.minimize(loss)
        tf.summary.scalar('loss_summary', loss)

        # 精度 todo
        precise = []
        for i in range(self.config.batch_size):
            precise.append(tf.cast(tf.equal(self.x, self.predict_y), dtype=tf.float32))
        tf.summary.scalar('precise_summary', tf.reduce_mean(precise))
        self.summary = tf.summary.merge_all()

    def encoder_frame(self, x, name):
        """
        :param x: [-1, 1701, 4]
        :param name: 
        :return: [-1, 1701, 10]
        """""
        with tf.variable_scope(name):
            x = tf.reshape(x, [-1, self.config.gene_length, self.config.num_chars, 1])  # [-1, 1701, 4, 1]

            ef_1 = tf.layers.conv2d(x, 32, (1, 4), 1, activation=tf.nn.relu, padding="valid", name="conv_1")  # [-1, 1701, 1, 32]
            ef_2 = tf.layers.conv2d(ef_1, 32, (3, 1), 3, activation=tf.nn.relu, padding="valid", name="conv_2")  # [-1, 567, 1, 32]
            ef_3 = tf.layers.conv2d(ef_2, 32, (3, 1), 3, activation=tf.nn.relu, padding="valid", name="conv_3")  # [-1, 189, 1, 32]

            f = tf.layers.flatten(ef_3)  # [-1, 189 * 32]

            vec = tf.layers.dense(f, 128, name="dense_1")
        return vec

    def decoder_frame(self, vec, name):
        """
        :param zf: [-1, 2048]
        :return:  [-1, 1701, 4]
        """""
        with tf.variable_scope(name):
            f = tf.layers.dense(vec, 189 * 32, name="dense_1")
            f = tf.reshape(f, [-1, 189, 1, 32])

            df_1 = tf.layers.conv2d_transpose(f, 32, (3, 1), (3, 1), activation=tf.nn.relu, padding="valid", name="conv_1")  # [-1, 567, 1, 32]
            df_2 = tf.layers.conv2d_transpose(df_1, 32, (3, 1), (3, 1), activation=tf.nn.relu, padding="valid", name="conv_2")  # [-1, 1701, 1, 32]
            df_3 = tf.layers.conv2d_transpose(df_2, 1, (1, 4), 1, padding="valid", name="conv_3")  # [-1, 1701, 4, 1]

            y = tf.reshape(df_3, [-1, 1701, 4])
        return y

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(shape=tf.shape(logvar), dtype=logvar.dtype)
            return mean + std * eps
        else:
            return mean + logvar * 0


class GeneApp:
    def __init__(self, config: GeneConfig):
        self.config = config
        graph = tf.Graph()  # 图
        with graph.as_default():
            self.ts = SubTensors(config)
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True  # 在没有gpu时使用cpu
            self.session = tf.Session(graph=graph, config=cfg)
            self.saver = tf.train.Saver()
            if config.new_model:
                self.session.run(tf.global_variables_initializer())
            else:
                try:
                    self.saver.restore(self.session, config.save_path)
                except:
                    self.session.run(tf.global_variables_initializer())

    def train(self, ds_train):
        cfg = self.config
        writer = tf.summary.FileWriter(logdir=cfg.logdir, graph=self.session.graph)
        batches = ds_train.num_examples // cfg.batch_size

        for epoch in range(cfg.epoches):
            for batch in range(batches):
                feed_dict = self.get_feed_dict(ds_train)
                _, summmary = self.session.run([self.ts.train_op, self.ts.summary], feed_dict)
                writer.add_summary(summmary, global_step=epoch * batches + batch)
                print("epoch = {epoch} , batch = {batch}".format(epoch=epoch, batch=batch))
                self.save()

    def get_feed_dict(self, ds):
        result = {self.ts.lr: self.config.lr}
        values = ds.next_batch(self.config.batch_size)
        for tensor, value in zip(self.ts.inputs, values):
            result[tensor] = value
        return result

    def test(self, ds_test):
        ds = self.config.ds
        datas = ds.next_batch(10)[-1]
        ts = self.ts

        y_predict = self.session.run(ts.predict_y, {ts.x: datas})[0]  # [1701]

        precise = 0
        x = datas[0]
        for i in range(self.config.gene_length):
            if x[i] == y_predict[i]:
                precise += 1
        print(float(precise)/self.config.gene_length)

        self.write_to_file(datas[0], y_predict)

    def write_to_file(self, H1N1_x, H1N1_y):
        """
        :param H1N1_x: [8, 1701]
        :param H1N1_y: [8, 1701]
        :return:
        """
        ds = self.config.ds
        file_path = "output.txt"
        file = open(file_path, "w")
        x = ds.to_gene(*H1N1_x)
        y = ds.to_gene(*H1N1_y)
        file.writelines(x)
        file.writelines("\n")
        file.writelines(y)
        file.close()

    def save(self):
        self.saver.save(self.session, save_path=self.config.save_path)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    config = GeneConfig()
    app = GeneApp(config)
    # app.train(config.ds)
    app.test(config.ds)
