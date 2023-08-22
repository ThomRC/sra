from sra-paper.models.losses import *

def select_loss_function(loss):
    smooth_loss = False
    margin_loss = False
    loss_func = None

    if loss.str.lower() == 'sce':
        loss_func = sce
    elif loss.str.lower() == 'mc_hinge':
        margin_loss = True
        loss_func = mc_hinge
    elif loss.str.lower() == 'margin':
        margin_loss = True
        loss_func = hinge_loss
    elif loss.str.lower() == 'sc-sce':
        ### 0) Smoothed classifier SCE
        smooth_loss = True
        loss_func = sc_sce
    elif loss.str.lower() == 'smc_hinge':
        smooth_loss = True
        margin_loss = True
        loss_func = smc_hinge
    elif loss.str.lower() == 'zhen':
        smooth_loss = True
        margin_loss = True
        loss_func = zhen
    ##### Different deployments of Smoothed Margin as loss function
    elif loss.str.lower() == 'sm1':
        smooth_loss = True
        loss_func = sm1_loss
    elif loss.str.lower() == 'clip_sm1':
        smooth_loss = True
        margin_loss = True
        loss_func = clip_sm1_loss
    elif loss.str.lower() == 'sm2':
        ### 1) "Naive" loss
        ### Just the second order sm approximation
        ### Needs the input variance as argument
        ### Goal: Increase the smoothed margin so that the smoothed classifier has a large margin
        ### Flaw: relatively low "clean" error
        ### Possible reason: the fitting of the smoothed classifier not necessarily guarantees the fitting of the
        ### "original" classifier
        smooth_loss = True
        loss_func = sm2
    elif loss.str.lower() == 'clip_sm2':
        ### 3) Clipped after shifting sm average loss
        ### max(0, d - sm1(x)) + max(0, d - sm2(x)) <- -(sm1(x) + sm2)
        ### Arguments: the large input variance (the small will be set as 1/100 of this value), a constant d that
        ### controls the smoothed margin size
        ### Goal: Instead of using the smoothed margin directly, which can lead to few samples increasing
        ### indefinitely, we add a constant and clip it, so that if sm > d, the gradient is zero and this terms
        ### stops contributing to the loss decrease
        ### Flaw: for d = 2 Linf adversarial robustness slightly better than the SCE loss
        ### Possible reasons:
        ### 1) d is too small and doesn't give enough room for the margin to increase
        ###     -> test with d = 4
        ### 2) the small variance margin can be much larger than the large variance one, dragging the loss to
        ### have larger low variance margin
        ### while most have small. For example, 10% having 10x larger margins than the other 90% (e.g. 3 and 0.3)
        ### is equivalent to all data having 0.57 margin
        smooth_loss = True
        margin_loss = True
        loss_func = clip_sm2_loss
    else:
        raise RuntimeError('Chosen loss is incorrect or not implemented \n '
                           'Please try again')
    print("Using {} loss".format(loss_func.str.lower()))
    return loss_func, smooth_loss, margin_loss