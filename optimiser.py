from typing_extensions import runtime

import numpy

import copy

import matplotlib

matplotlib.use('Agg')

matplotlib.rcParams['agg.path.chunksize'] = int(1e16)

import matplotlib.pyplot

import datetime

numpy.set_printoptions(precision=2)


class ADAM:

    def __init__(self, theta, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8):

        self.lr = lr

        self.beta_1 = beta_1

        self.beta_2 = beta_2

        self.eps = eps

        self.m = numpy.zeros(numpy.shape(theta))

        self.v = numpy.zeros(numpy.shape(theta))

    def update(self, theta, g_t):

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * g_t

        self.v = self.beta_2 * self.v + (1 - self.beta_2) * g_t * g_t

        theta = theta - self.lr * self.m / (self.v ** 0.5 + self.eps)

        return theta


def parameter_update(theta_0, data, extra_args, obj, obj_g, optimiser_choice='adam',
                     lr=1e-3, batch_size=32, val_size=None, val_skip=0, tol_r=10, plot_tol_r=1,
                     factr=1e-3, max_batch_r=None,
                     early_stop=True,
                     ref='tmp_mdl',
                     plot_loss=False, print_info=True, plot_final_loss=False, print_iteration=False, print_final_info=True):

    start_time = datetime.datetime.now()
    
    raw_batch_L = []
    
    epoch_L = []

    gap = []

    theta = copy.deepcopy(theta_0)

    fin_theta = copy.deepcopy(theta_0)

    n_data = numpy.shape(data)[0]
    
    epoch_size = int(numpy.ceil(n_data / batch_size)) 
    
    tol = int(epoch_size * tol_r)
        
    plot_tol = int(epoch_size * plot_tol_r)
        
    max_batch = int(numpy.ceil(n_data / batch_size) * max_batch_r)

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
            if batch_size >= n_data:
                batch_size = n_data
                batch_idx = numpy.arange(0, n_data)
            else:    
                batch_idx = numpy.random.choice(numpy.arange(0, n_data), batch_size, replace=False)

        L_t, g_t = obj_g(theta, data[batch_idx, :], extra_args)
        
        if numpy.sum(~numpy.isfinite(g_t)) == 0:
            theta = optimiser.update(theta, g_t)
        else:
            raise RuntimeError('nan in gradient')
        
        if numpy.isfinite(L_t.numpy()):
            L_t = L_t.numpy()
        else:
            raise RuntimeError('nan in Loss')

        if val_size is not None:

            if numpy.mod(i, numpy.min([numpy.floor(tol / 2), (1 + val_skip)])) == 0:

                L_t = obj(theta,  data[val_idx, :], extra_args)

            else:

                L_t = copy.deepcopy(raw_batch_L[-1])

            if numpy.isfinite(L_t.numpy()):
                L_t = L_t.numpy()
            else:
                raise RuntimeError('nan in Loss')
        
        if print_iteration:        
            print('Batch: ' + str(i) + ', L_t: ' + str(L_t))
                
        raw_batch_L.append(L_t)
                
        if len(raw_batch_L) > epoch_size:
            epoch_L.append(numpy.mean(numpy.array(raw_batch_L)[-epoch_size:]))
        else:
            epoch_L.append(numpy.mean(numpy.array(raw_batch_L)))

        if (numpy.mod(len(epoch_L), plot_tol) == 0) & plot_loss & (len(epoch_L) >= plot_tol):

            fig, axlist = matplotlib.pyplot.subplots(nrows=1, ncols=2, dpi=128, figsize=(21, 9))

            axlist[0].plot(numpy.arange(len(epoch_L)), numpy.array(epoch_L), 'b', alpha=1.0)
            axlist[0].plot(numpy.arange(len(raw_batch_L)), numpy.array(raw_batch_L), 'b', alpha=0.1)
            
            v_max = numpy.percentile(numpy.array(epoch_L), 95)
            v_min = numpy.min(epoch_L)
            
            # axlist[0].set_ylim([v_min - 0.01 * (v_max - v_min), v_max + 0.01 * (v_max - v_min)])

            axlist[0].set_xlabel('Batches')

            axlist[0].set_ylabel('Loss')

            axlist[0].set_title('Learning Rate: ' + str(lr))

            axlist[0].grid(True)
        
            axlist[1].plot(numpy.arange(len(epoch_L))[-tol:], numpy.array(epoch_L)[-tol:], 'b', alpha=1.0)
            axlist[1].plot(numpy.arange(len(raw_batch_L))[-tol:], numpy.array(raw_batch_L)[-tol:], 'b', alpha=0.1)

            axlist[1].set_xlabel('Batches')

            axlist[1].set_ylabel('Loss')

            axlist[1].set_title('Learning Rate: ' + str(lr))

            axlist[1].grid(True)

            try:
                fig.savefig('./' + ref + str(lr) + '_' + str(numpy.shape(data)[0]) + '_' 
                            + str(start_time).replace(':', '-') + '.png', bbox_inches='tight')
            except PermissionError:
                pass
            except OSError:
                pass

            matplotlib.pyplot.close(fig)
            
            try:
                numpy.savetxt(fname='./' + ref + str(lr) + '_' + str(numpy.shape(data)[0]) + '_' + 
                                                str(start_time).replace(':', '-') + '.csv', X=fin_theta, delimiter=',')
            except PermissionError:
                pass
            except OSError:
                pass
            
        if len(epoch_L) > (2 * tol):
            previous_opt = numpy.mean(epoch_L.copy()[-2*tol:-tol])

            current_opt = numpy.mean(epoch_L.copy()[-tol:])

            gap.append(previous_opt - current_opt)
            
            if current_opt < previous_opt:
                fin_theta = copy.deepcopy(theta)
                
            if len(gap) >= 3:
                if (gap[-1] < numpy.clip(gap[-2] * factr, 0.0, None)) & early_stop:
                    break
            
        if print_info: 
            
            tmp_time = datetime.datetime.now()
            
            if len(epoch_L) <= (3 * tol):
                print('\rEpoch: ' + str(int(len(epoch_L) / epoch_size)) + ', Optimiser: ' + optimiser_choice + 
                      ', Loss: ' + "{:.6f}".format(epoch_L[-1]) + 
                      ', Progress:' + "{:.2f}".format(len(raw_batch_L) / max_batch * 100) + '%' + 
                     ', Running Time: ' + str(((tmp_time - start_time)))[:7] + 
                      ', Remaining Time: ' + str((tmp_time - start_time) * (max_batch / len(raw_batch_L) - 1))[:7], end='')

            else:
                print('\rEpoch: ' + str(int(len(epoch_L) / epoch_size)) + ', Optimiser: ' + optimiser_choice + 
                      ', Loss: ' + "{:.6f}".format(epoch_L[-1]) + 
                      ', Progress:' + "{:.2f}".format(len(raw_batch_L) / max_batch * 100) + '%' + 
                      ', Previous Avg.Loss:' + "{:.6f}".format(previous_opt) +
                      ', Current Avg.Loss:' + "{:.6f}".format(current_opt) +
                      ', Improvement: ' + "{:.6f}".format(gap[-1]) + 
                      ', Threshold: ', "{:.6f}".format(numpy.clip(gap[-2] * factr, 0.0, None) ) +
                      ', Running Time: ' + str(((tmp_time - start_time)))[:7] + 
                      ', Remaining Time: ' + str((tmp_time - start_time) * (max_batch / len(raw_batch_L) - 1))[:7], end='')                  
                 
            
    if print_final_info:
        print('\nTotal epoch number: ' + str(int((len(epoch_L) / epoch_size))))
        print('Initial Loss: ' + str(epoch_L[0]))
        print('Final Loss: ' + str(numpy.mean(epoch_L.copy()[-tol:])))
        print('Current Improvement, Previous Improvement * factr')
        print(numpy.hstack([gap[-1], numpy.clip(gap[-2] * factr, 0.0, None)]))

    if plot_final_loss:

        fig, axlist = matplotlib.pyplot.subplots(nrows=1, ncols=2, dpi=128, figsize=(21, 9))
        
        axlist[0].plot(numpy.arange(len(epoch_L)), numpy.array(epoch_L), 'b', alpha=1.0)
        axlist[0].plot(numpy.arange(len(raw_batch_L)), numpy.array(raw_batch_L), 'b', alpha=0.1)
            
        v_max = numpy.percentile(numpy.array(epoch_L), 95)
        v_min = numpy.min(epoch_L)
            
        # axlist[0].set_ylim([v_min - 0.01 * (v_max - v_min), v_max + 0.01 * (v_max - v_min)])

        axlist[0].set_xlabel('Batches')

        axlist[0].set_ylabel('Loss')

        axlist[0].set_title('Learning Rate: ' + str(lr))

        axlist[0].grid(True)
        
        axlist[1].plot(numpy.arange(len(epoch_L))[-tol:], numpy.array(epoch_L)[-tol:], 'b', alpha=1.0)
        axlist[1].plot(numpy.arange(len(raw_batch_L))[-tol:], numpy.array(raw_batch_L)[-tol:], 'b', alpha=0.1)

        axlist[1].set_xlabel('Batches')

        axlist[1].set_ylabel('Loss')

        axlist[1].set_title('Learning Rate: ' + str(lr))

        axlist[1].grid(True)
        
    # if not early_stop:
    #     fin_theta = copy.deepcopy(theta)

    return fin_theta
