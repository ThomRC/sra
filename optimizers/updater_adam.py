import typing as tp  # NOQA

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, clip = 1., schedule = 'cst', lr_0 = 1.0, **kwargs):
        super(Adam, self).__init__()
        self.lr = lr
        self.lr_sch = lr_0
        # self.lr_sch_epoch = 30
        self.lr_sch_epoch = 20
        # self.lr_decay_counter = self.xp.ceil(self.lr_decay_epoch * 468.75)
        # 468.75 iterations = 1 epoch for 60000 training data and 128 minibatch
        # 390.625 iterations = 1 epoch for 50000 training data and 128 minibatch
        self.beta1 = beta1
        self.beta2 = beta2
        self.clip = clip
        self.counter = 0
        self.counter_bu = 0
        self.m = None
        self.v = None
        self.m_bu = None
        self.v_bu = None
        self.reg_grad = 0
        self.adam_grad = 0
        self.schedule = schedule
        self.last_epoch = 0
        self.epoch_counter = 0
        self.step_counter = 0
        self.step_counter_bu = 0
        self.spe = 0 ### step per epoch
        self.W_bu = self.W.array

    def mean_abs_grad(self):
        # Returns the mean absolute gradient for respective layer
        loss_grad = self.xp.mean(self.xp.absolute(self.W.grad))
        return loss_grad

    def update(self, epoch, grad_norm):
        if self.m is None:
            self.m = self.xp.zeros_like(self.W.array)
            self.v = self.xp.zeros_like(self.W.array)

        if epoch == 0:
            self.spe += 1
        else:
            if self.counter % self.spe == 0:
                self.W_bu = self.xp.copy(self.W.array)
                self.m_bu = self.xp.copy(self.m)
                self.v_bu = self.xp.copy(self.v)

            if self.last_epoch != epoch:
                if self.last_epoch == 0:
                    self.step_counter += self.spe

                self.last_epoch = epoch
                self.epoch_counter += 1

            if self.schedule == 'cos-ann':
                self.lr_sch = 0.5 * (1 + self.xp.cos(self.xp.pi * self.step_counter / (self.spe * self.lr_sch_epoch))).astype('float32')

                if self.epoch_counter == self.lr_sch_epoch:
                    self.step_counter = 0
                    self.epoch_counter = 0
                    print("### Warm Restart: {} epochs".format(self.lr_sch_epoch))
                    self.lr_sch_epoch *= 2
            elif self.schedule == 'cst':
                pass
            elif self.schedule == 'step':
                if epoch % self.lr_sch_epoch == 0:
                    self.lr_sch *= 0.1
                    print("################################ lr = {}".format(self.lr_sch))
            else:
                raise RuntimeError('Chosen scheduling is incorrect or non-existent. \n '
                                   'Currently implemented: '
                                   '\n- Constant (\'cst\')'
                                   '\n- Step decay (\'step\')'
                                   '\n- Cosine annealing (\'cos-ann\')')
            self.step_counter += 1

        if self.xp.isnan(self.W.grad.max()):
            if self.counter > 2000:
                print("Gradient failure. Reducing lr and restoring weights from {} previous steps".format(self.spe))
                self.W.array = self.xp.copy(self.W_bu)
                self.m = self.xp.copy(self.m_bu)
                self.v = self.xp.copy(self.v_bu)
                self.lr = self.lr/10
                print(self.lr)
            return 1
        else:
            norm_ratio = 1 / (grad_norm + 0.0001)

            adam_mult = (self.xp.sqrt(1.0 - self.beta2**self.counter) / (1.0 - self.beta1**self.counter)).astype('float32')
            self.m += (1 - self.beta1) * ((norm_ratio * self.W.grad) - self.m)
            self.v += (1 - self.beta2) * ((norm_ratio * self.W.grad) ** 2 - self.v)
            self.adam_grad = adam_mult * self.m / (self.xp.sqrt(self.v) + 1e-15)

            if self.counter > 2000:
                self.W.array -= self.lr_sch * self.lr * self.adam_grad

            self.counter += 1

            return 1