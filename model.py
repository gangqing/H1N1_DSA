from tensorflow.compat import v1 as tf
from H1N1_DSA.GeneData import Data
import H1N1_DSA.framework as fm


class MyTensors(fm.Tensors):
    def compute_grads(self, opt):
        vars = tf.trainable_variables()
        grads = []
        for gpu_id, ts in enumerate(self.sub_ts):
            with tf.device('/gpu:%d' % gpu_id):
                grads.append([opt.compute_gradients(ts.losses[0], vars)])
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]


class GeneConfig(fm.Config):
    def __init__(self):
        super().__init__()
        self.china_path = "data/GeneFastaResults_China_4_8_2020.fasta"
        self.canada_path = "data/GeneFastaResults_Canada_4_8_2020.fasta"
        self.australia_path = "data/GeneFastaResults_Australia_5_8_2020.fasta"
        self.hongkong_path_2009 = "data/GeneFastaResults_HongKong_2009.fasta"
        self.kenya_path_2009 = "data/GeneFastaResults_Kenya.fasta"
        self.egypt_2009_path = "data/GeneFastaResults_ Egypt_2009.fasta"
        self.guam_2009_path = "data/GeneFastaResults_Guam_2009.fasta"
        self.iran_2009_2010_path = "data/GeneFastaResults_lran_2009-2010.fasta"
        self.guam_2009_txt_path = "data/archaeopteryx_2009_Guam.txt"
        self.num_chars = 4
        self.ds = None
        self.batch_size = 4
        self.new_model = True
        self.epoches = 20
        self.lr = 0.0001
        self.num_frame = 8
        self.f_units = 256
        self.z_units = 32
        self.keep_prob = 0.8
        # todo
        self.gene_length = 1701  # 一个样本的基因长度
        self.training = True

    def get_name(self):
        return "gene"

    def get_sub_tensors(self, gpu_index):
        return SubTensors(self)

    def get_app(self):
        return GeneApp(self)

    def get_ds_train(self):
        self.read_ds()
        return self.ds

    def get_ds_test(self):
        self.read_ds()
        return self.ds
    
    def test(self):
        self.keep_prob = 1.0
        self.training = False
        super(GeneConfig, self).test()

    def read_ds(self):
        if self.ds is None:
            self.ds = Data(self)

    def get_tensors(self):
        return MyTensors(self)


class SubTensors:
    def __init__(self, config: GeneConfig):
        self.config = config
        # 训练网络
        self.x = tf.placeholder(tf.int64, [None, config.num_frame, config.gene_length], name="x")  # [-1, 8, 1701]
        self.inputs = [self.x]
        x = tf.one_hot(self.x, config.num_chars, dtype=tf.float32)  # [-1, 8, 1701, 4]
        y = tf.layers.dense(x, 10, activation=tf.nn.relu, name="dense_1")  # [-1, 8, 1701, 10]
        vec = self.encoder_frame(y, "encoder_frame")  # [-1, 8, 2048]

        self.mean_f, self.logvar_f = self.encoder_f(vec, "encoder_f")  # [-1, 256]
        self.f = self.reparameterize(self.mean_f, self.logvar_f, self.config.training)
        f = tf.reshape(self.f, [-1, 1, self.f.shape[-1]])
        f = tf.tile(f, [1, config.num_frame, 1])  # [-1, 8, 256]

        self.mean_z, self.logvar_z = self.encoder_z(vec, f, "encoder_z")  # [-1, 8, 32]
        self.z = self.reparameterize(self.mean_z, self.logvar_z, self.config.training)

        zf = tf.concat((self.z, f), axis=2)  # [-1, 8, 256 + 32]
        logits = self.decoder_frame(zf, "decoder_frame")  # [-1, 8, 1701, 4]
        # logits = tf.layers.dense(zf, config.gene_length * config.num_chars, name="dense_2")  # [-1, 8, 1701 * 4]
        # logits = tf.reshape(logits, [-1, 8, 1701, config.num_chars])
        self.predict_y = tf.argmax(logits, axis=3)  # [-1, 8, 1701]

        self.loss(x, logits)

    def loss(self, x, logits):
        """
        :param x: [-1, 8, 1701, 4]
        :param logits: [-1, 8, 1701, 4]
        :return:
        """""
        softmax = tf.nn.softmax_cross_entropy_with_logits_v2(x, logits, axis=3)  # [-1, 8, 1701]
        mes = tf.reduce_mean(tf.reduce_sum(softmax, axis=2))
        # mes = tf.reduce_mean(softmax)
        # f : [-1, 256]
        # kld_f = tf.reduce_mean(
        #     -0.5 * tf.reduce_sum(1 + self.logvar_f - tf.pow(self.mean_f, 2) - tf.exp(self.logvar_f), 1))

        loss = mes
        self.losses = [loss, mes]

    def encoder_frame(self, x, name):
        """
        :param x: [-1, 8, 1701, 4]
        :param name: 
        :return: [-1, 8, 2048]
        """""
        with tf.variable_scope(name):
            ef_0 = tf.reshape(x, [-1, self.config.gene_length, 10, 1])  # [-1, 1701, 4, 1]

            ef_0 = tf.layers.conv2d(ef_0, 8, (1, 10), 1, padding="valid", name="conv_1")  # [-1, 1701, 1, 8]

            ef_1 = tf.layers.conv2d(ef_0, 16, (3, 1), 3, activation=tf.nn.relu, padding="valid", name="conv_2")  # [-1, 567, 1, 16]

            ef_2 = tf.layers.conv2d(ef_1, 32, (3, 1), 3, activation=tf.nn.relu, padding="valid", name="conv_3")  # [-1, 189, 1, 32]
            ef_2 = tf.layers.flatten(ef_2)  # [-1, 189 * 32] : [-1, 6048]

            ef_3 = tf.layers.dense(ef_2, 2048, activation=tf.nn.relu, name="dense_1")  # [-1, 2048]
            ef_4 = tf.reshape(ef_3, [-1, 8, 2048])
        return ef_4

    def encoder_f(self, vec, name):
        """
        :param vec: [-1, 8, 2048]
        :param name: 
        :return: [-1, 256]
        """""
        with tf.variable_scope(name):
            x = tf.layers.dense(vec, self.config.f_units, name="dense1")  # [-1, 8, 256]

            cell_l2r = tf.nn.rnn_cell.LSTMCell(self.config.f_units, name="cell_l2r", state_is_tuple=False)
            cell_r2l = tf.nn.rnn_cell.LSTMCell(self.config.f_units, name="cell_r2l", state_is_tuple=False)
            batch_size = tf.shape(x)[0]
            state_l2r = cell_l2r.zero_state(batch_size, dtype=x.dtype)
            state_r2l = cell_r2l.zero_state(batch_size, dtype=x.dtype)
            for i in range(self.config.num_frame):
                y_l2r, state_l2r = cell_l2r(x[:, i, :], state_l2r)  # y_l2r : [-1, 256]
                y_r2l, state_r2l = cell_r2l(x[:, self.config.num_frame - i - 1, :], state_r2l)  # y_r2l : [-1, 256]

            y = tf.concat((y_l2r, y_r2l), axis=1)  # [-1, 1024]
            mean_f = tf.layers.dense(y, self.config.f_units, name="dense_mean")  # [-1, 256]
            logvar_f = tf.layers.dense(y, self.config.f_units, name="dense_logvar")  # [-1, 256]
        return mean_f, logvar_f

    def encoder_z(self, vec, f, name):
        """ 
        :param x: [-1, 8, 2048]
        :param f: [-1, 8, 256]
        :param name: 
        :return: [-1, 8, 32]
        """""
        with tf.variable_scope(name):
            x = tf.concat((vec, f), axis=2)  # [-1, 8, 2048 + 256]
            num_units = 512
            x = tf.layers.dense(x, num_units, name="dense_2")  # [-1, 8, 512]

            cell_l2r = tf.nn.rnn_cell.LSTMCell(num_units, name="cell_l2r", state_is_tuple=False)
            cell_r2l = tf.nn.rnn_cell.LSTMCell(num_units, name="cell_r2l", state_is_tuple=False)
            # cell_rnn = tf.nn.rnn_cell.LSTMCell(self.config.f_units, name="cell_rnn", state_is_tuple=False)
            batch_size = tf.shape(x)[0]
            state_l2r = cell_l2r.zero_state(batch_size, dtype=x.dtype)
            state_r2l = cell_r2l.zero_state(batch_size, dtype=x.dtype)
            # state_rnn = cell_rnn.zero_state(batch_size, dtype=x.dtype)
            y_l2r = []
            y_r2l = []  # [8, -1, 512]
            for i in range(self.config.num_frame):
                yi_l2r, state_l2r = cell_l2r(x[:, i, :], state_l2r)  # y_l2r : [-1, 512]
                yi_r2l, state_r2l = cell_r2l(x[:, self.config.num_frame - i - 1, :], state_r2l)  # y_r2l : [-1, 512]
                y_l2r.append(yi_l2r)
                y_r2l.insert(0, yi_r2l)
            y_lstm = [yi_l2r + yi_r2l for yi_l2r, yi_r2l in zip(y_l2r, y_r2l)]  # [8, -1, 512]
            y_lstm = tf.transpose(y_lstm, [1, 0, 2])  # [-1, 8, 512]
            # y_rnn = []
            # for i in range(self.config.num_frame):
            #     yi_rnn, state_rnn = cell_rnn(y_lstm[:, i, :], state_rnn)  # [-1, 512]
            #     y_rnn.append(yi_rnn)
            # features = tf.transpose(y_rnn, [1, 0, 2])
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.f_units * 2)
            features, states = tf.nn.dynamic_rnn(rnn_cell, y_lstm, dtype=tf.float32)  # [-1, 8, 512]
            mean_z = tf.layers.dense(features, self.config.z_units, name="dense_mean")  # [-1, 8, 32]
            logvar_z = tf.layers.dense(features, self.config.z_units, name="dense_logvar")  # [-1, 8, 32]
        return mean_z, logvar_z

    def simple_z(self, batch_size, name):
        """
        :return: [-1, 8, 32]
        """
        z_out = None
        z_mean = None
        z_logvar = None
        num_units = self.config.z_units
        with tf.variable_scope(name) as scope:
            cell = tf.nn.rnn_cell.LSTMCell(num_units, name="cell", state_is_tuple=False)
            state = cell.zero_state(batch_size, dtype=self.z.dtype)
            x_t = tf.zeros([batch_size, num_units], dtype=self.z.dtype)
            for i in range(self.config.num_frame):
                h_t, state = cell(x_t, state)  # [-1, 32]
                mean_z = tf.layers.dense(h_t, num_units, name="mean_z")
                logvar_z = tf.layers.dense(h_t, num_units, name="logvar_z")
                z_t = self.reparameterize(mean_z, logvar_z)
                if z_mean is None:
                    z_mean = tf.reshape(mean_z, [-1, 1, num_units])
                    z_logvar = tf.reshape(logvar_z, [-1, 1, num_units])
                    z_out = tf.reshape(z_t, [-1, 1, num_units])
                else:
                    z_mean = tf.concat((z_mean, tf.reshape(mean_z, [-1, 1, num_units])), axis=1)
                    z_logvar = tf.concat((z_logvar, tf.reshape(logvar_z, [-1, 1, num_units])), axis=1)
                    z_out = tf.concat((z_out, tf.reshape(z_t, [-1, 1, num_units])), axis=1)
                scope.reuse_variables()

        return z_mean, z_logvar, z_out

    def decoder_frame(self, zf, name):
        """
        :param zf: [-1, 8, 256 + 32]
        :return:  [-1, 8, 1701, 4]
        """""
        with tf.variable_scope(name):
            shape = zf.shape[-1]
            df_0 = tf.reshape(zf, [-1, shape])  # [-1, 32 + 256]
            df_1 = tf.layers.dense(df_0, 2048, name="dense_1")  # [-1, 2048]

            df_2 = tf.layers.dense(df_1, 189 * 32, activation=tf.nn.relu, name="dense_2")  # [-1, 189 * 32]
            df_2 = tf.reshape(df_2, [-1, 189, 1, 32])  # [-1, 189, 1, 32]

            df_3 = tf.layers.conv2d_transpose(df_2, 16, (3, 1), (3, 1), activation=tf.nn.relu, padding="same", name="conv_1")  # [-1, 567, 1, 16]
            df_4 = tf.layers.conv2d_transpose(df_3, 8, (3, 1), (3, 1), activation=tf.nn.relu, padding="same", name="conv_2")  # [-1, 1701, 1, 8]
            df_5 = tf.layers.conv2d_transpose(df_4, 1, (1, 4), 1, padding="valid", name="conv_3")  # [-1, 1701, 4, 1]
            df_6 = tf.reshape(df_5, [-1, 8, 1701, 4])

        return df_6

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(shape=tf.shape(logvar), dtype=logvar.dtype)
            return mean + std * eps
        else:
            return mean + logvar * 0


class GeneApp(fm.App):
    def __init__(self, config: GeneConfig):
        super().__init__(config)

    def test(self, ds_test):
        ds = self.config.ds
        datas = ds.next_batch(1)[0]
        ts = self.ts.sub_ts[-1]

        y_predict = self.session.run(ts.predict_y, {ts.x: datas})[0]
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
        for i in range(self.config.num_frame):
            x = ds.to_gene(*H1N1_x[i])
            y = ds.to_gene(*H1N1_y[i])
            file.writelines(x)
            file.writelines("\n")
            file.writelines(y)
            file.writelines("\n\n")
        file.close()


if __name__ == '__main__':
    config = GeneConfig()
    # config.from_cmd()
    config.call("test")
