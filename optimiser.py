import numpy

import copy

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot

import datetime


class ADAM:

    def __init__(self, theta, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8):

        self.lr = lr

        self.beta_1 = beta_1

        self.beta_2 = beta_2

        self.eps = eps

        self.m = numpy.zeros(numpy.shape(theta))

        self.v = numpy.zeros(numpy.shape(theta))

    def update(self, theta, g_t):

        if numpy.sum(numpy.isfinite(g_t)) == len(g_t):

            self.m = self.beta_1 * self.m + (1 - self.beta_1) * g_t

            self.v = self.beta_2 * self.v + (1 - self.beta_2) * g_t * g_t

            theta = theta - self.lr * self.m / (self.v ** 0.5 + self.eps)

        return theta


def parameter_update(theta_0, data, extra_args, obj, obj_g, optimiser_choice='adam',
                     lr=1e-3, batch_size=None, val_size=None, val_skip=0, tol=8, factr=1e-3, max_batch=int(1e4),
                     plot_loss=True, print_info=True, plot_final_loss=True):

    batch_L = []

    gap = []

    theta = copy.deepcopy(theta_0)

    fin_theta = copy.deepcopy(theta_0)

    n_data = numpy.shape(data)[0]

    if val_size is not None:
        if val_size >= n_data:
            val_size = n_data
            val_idx = numpy.arange(0, n_data)
        else:
            val_idx = numpy.random.choice(numpy.arange(0, n_data), val_size, replace=False)
    else:
        val_size = None

    batch_idx = numpy.arange(0, n_data)

    if optimiser_choice == 'adam':
        optimiser = ADAM(theta)
    else:
        raise NotImplementedError

    for i in range(0, max_batch):

        if batch_size is not None:
            batch_idx = numpy.random.choice(numpy.arange(0, n_data), batch_size, replace=False)

        L_t, g_t = obj_g(theta, data[batch_idx, :], extra_args)

        theta = optimiser.update(theta, g_t)

        if val_size is not None:

            if numpy.mod(i, numpy.min([numpy.floor(tol / 2), (1 + val_skip)])) == 0:

                L_t = obj(theta,  data[val_idx, :], extra_args)

            else:

                L_t = copy.deepcopy(batch_L[-1])

        batch_L.append(L_t)

        if len(batch_L) >= 2:
            if batch_L[-1] < numpy.min(batch_L[:-1]):
                fin_theta = copy.deepcopy(theta)

        if (numpy.mod(len(batch_L), tol) == 0) & plot_loss:

            fig = matplotlib.pyplot.figure(figsize=(16, 9))

            matplotlib.pyplot.plot(numpy.arange(0, len(batch_L)),
                                   numpy.array(batch_L))

            matplotlib.pyplot.xlabel('Batches')

            matplotlib.pyplot.ylabel('Loss')

            matplotlib.pyplot.title('Learning Rate: ' + str(lr))

            matplotlib.pyplot.grid(True)

            try:
                fig.savefig('./' + str(lr) + '.png', bbox_inches='tight')
            except PermissionError:
                pass
            except OSError:
                pass

            matplotlib.pyplot.close(fig)

        if len(batch_L) > tol:
            previous_opt = numpy.min(batch_L.copy()[:-tol])

            current_opt = numpy.min(batch_L.copy()[-tol:])

            gap.append(previous_opt - current_opt)

            if (numpy.mod(len(batch_L), tol) == 0) & print_info:
                print('=============================================================================')

                print('Batch: ' + str(len(batch_L)) + ', optimiser: ' + optimiser_choice + ', Loss: ' + str(L_t))

                print('Previous And Recent Top Averaged Loss Is:')
                print(numpy.hstack([previous_opt, current_opt]))

                print('Current Improvement, Initial Improvement * factr')
                print(numpy.hstack([gap[-1], gap[0] * factr]))

                print('=============================================================================')

            if (len(gap) >= 2) & (gap[-1] <= (gap[0] * factr)):
                print('Total batch number: ' + str(len(batch_L)))
                print('Initial Loss: ' + str(batch_L[0]))
                print('Final Loss: ' + str(numpy.min(batch_L)))
                print('Current Improvement, Initial Improvement * factr')
                print(numpy.hstack([gap[-1], gap[0] * factr]))
                break

    if plot_final_loss:

        fig = matplotlib.pyplot.figure(figsize=(16, 9))

        matplotlib.pyplot.plot(numpy.arange(0, len(batch_L)),
                               numpy.array(batch_L))

        matplotlib.pyplot.xlabel('Batches')

        matplotlib.pyplot.ylabel('Loss')

        matplotlib.pyplot.title('Learning Rate: ' + str(lr))

        matplotlib.pyplot.grid(True)

        try:
            fig.savefig('./' + str(lr) + '.png', bbox_inches='tight')
        except PermissionError:
            pass
        except OSError:
            pass

        matplotlib.pyplot.close(fig)

    return fin_theta
