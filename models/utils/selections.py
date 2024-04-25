from models.losses import *

def select_loss_function(loss):
    """
    Function that returns the loss function to be used as well as what loss hyperparameters it needs (x_var and d)

    Args:
        loss: string with the name of the loss to be used on training

    Returns:
        loss_func: loss function from the losses.py file
        smooth_loss: whether this loss requires input smoothing by adding isotropic Gaussian noise with parameter x_var
        margin_loss: whether this loss requires a maximum margin control parameter d
    """
    smooth_loss = False
    margin_loss = False

    if loss == 'sce':
        # Softmax Cross-Entropy (SCE)
        loss_func = sce
    elif loss == 'mh':
        # Multiclass Hinge loss
        margin_loss = True
        loss_func = mc_hinge
    elif loss == 'mcr':
        # Margin Certified Radius
        margin_loss = True
        loss_func = mcr_loss
    elif loss == 'sc-sce':
        # Smoothed Classifier SCE
        smooth_loss = True
        loss_func = sc_sce
    elif loss == 'zhen':
        # Zhen loss (Zhen et al. (2021)  Simpler Certified Radius Maximization by Propagating Covariances.)
        smooth_loss = True
        margin_loss = True
        loss_func = zhen_loss
    elif loss == 'mscr':
        # Margin Smoothed Certified Radius
        smooth_loss = True
        margin_loss = True
        loss_func = mscr_loss
    else:
        raise RuntimeError('Chosen loss is incorrect or not implemented \n '
                           'Please try again')
    print("Using {} loss".format(loss))
    return loss_func, smooth_loss, margin_loss