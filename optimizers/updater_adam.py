import typing as tp  # NOQA
import numpy as np

class Adam:
    """ Implementation of Adam with warm-restart """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, clip = 1., schedule = 'cst', lr_0 = 1.0, **kwargs):
        super(Adam, self).__init__()
        self.lr = lr
        self.lr_sch = lr_0
        self.lr_sch_epoch = 20 # When = 20 in 300 epochs it finishes the fourth warm restart cycle
        self.adam_mult = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.clip = clip
        self.counter = 0
        self.counter_bu = 0
        self.m = None
        self.v = None
        self.m_bu = None
        self.v_bu = None
        self.adam_grad = 0
        self.schedule = schedule
        self.last_epoch = 0
        self.epoch_counter = 0
        self.step_counter = 0
        self.step_counter_bu = 0
        self.spe = 0 ### step per epoch
        self.W_bu = self.W.array
        self.upd_start = 2000

    def update(self, epoch, grad_norm):
        if self.m is None:
            # If first iteration it creates the arrays for the m and v variables required for Adam
            self.m = self.xp.zeros_like(self.W.array)
            self.v = self.xp.zeros_like(self.W.array)

        if epoch == 0:
            self.spe += 1 # counts the number of steps for the first epoch, to be used in the remaining of the updater
        else:
            if self.counter % self.spe == 0:
                # Stores the current weights, m and v variables for current layer every time a new epoch starts
                self.W_bu = self.xp.copy(self.W.array)
                self.m_bu = self.xp.copy(self.m)
                self.v_bu = self.xp.copy(self.v)

            if self.last_epoch != epoch:
                # Used in case there is an error in the gradient and the previous epoch backup is used
                if self.last_epoch == 0:
                    self.step_counter += self.spe

                self.last_epoch = epoch
                self.epoch_counter += 1

            if self.schedule == 'cos-ann':
                # Learning rate decay for cosine annealing
                self.lr_sch = 0.5 * (1 + np.cos(np.pi * self.step_counter / (self.spe * self.lr_sch_epoch))).astype('float32')

                if self.epoch_counter == self.lr_sch_epoch:
                    self.step_counter = 0
                    self.epoch_counter = 0
                    print("### Warm Restart: {} epochs".format(self.lr_sch_epoch))
                    self.lr_sch_epoch *= 2
            elif self.schedule == 'cst':
                # Constant learning rate
                pass
            elif self.schedule == 'step':
                # Step learning rate decay
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
            print('Nan in grad')
            # In case there's any nan in the gradients it restores the weights, m and v from the last epoch's backup
            if self.counter > self.upd_start:
                print("Gradient failure. Reducing lr and restoring weights from {} previous steps".format(self.spe))
                self.W.array = self.xp.copy(self.W_bu)
                self.m = self.xp.copy(self.m_bu)
                self.v = self.xp.copy(self.v_bu)
                self.lr = self.lr/10
                print(self.lr)
        else:
            # Computes the Adam update
            # Gradient normalization factor
            norm_ratio = 1 / (grad_norm + 0.0001)
            # norm_ratio = 1

            self.m += (1 - self.beta1) * ((norm_ratio * self.W.grad) - self.m)
            self.v += (1 - self.beta2) * ((norm_ratio * self.W.grad) ** 2 - self.v)
            
            # Updates after minimal number of steps
            if self.counter > self.upd_start:
                self.adam_mult = (np.sqrt(1.0 - self.beta2**(self.counter-self.upd_start)) / (1.0 - self.beta1**(self.counter-self.upd_start))).astype('float32')
                self.adam_grad = self.adam_mult * self.m / (self.xp.sqrt(self.v + 1e-15))
                self.W.array -= self.lr_sch * self.lr * self.adam_grad
            
            self.counter += 1
