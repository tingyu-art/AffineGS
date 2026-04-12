import numpy as np
from skimage.metrics import structural_similarity

def RMSE(ut, vt, um, vm, percentile=None):
    """
    1. Root Mean Square ERROR
    Ref: https://doi.org/10.1007/s00348-020-03062-x
    """
    if percentile is not None:
        errors = (ut - um)**2 + (vt - vm)**2
        # 按照误差值排序，并去掉最大和最小的 5% 的值
        lower_percentile = np.percentile(errors, percentile)
        upper_percentile = np.percentile(errors, 100 - percentile)
        # 过滤掉小于 lower_percentile 或大于 upper_percentile 的值
        filtered_errors = errors[(errors >= lower_percentile) & (errors <= upper_percentile)]
        # 计算剩余数据的平均值
        mean_error = np.mean(filtered_errors)
        # 计算RMSE
        rmse = np.sqrt(mean_error)
    else:
        rmse = np.sqrt(np.mean((ut-um)**2+(vt-vm)**2))
    return rmse

def EPE(ut, vt, um, vm):
    """End-Point Error"""
    epe = np.sqrt((um-ut)**2+(vm-vt)**2)
    return epe

def AEE(ut, vt, um, vm, percentile=None):
    """
    2. Average End-Point Error
    Ref: https://doi.org/10.1038/s42256-021-00369-0
    """
    if percentile is not None:
        errors = np.sqrt((ut-um)**2+(vt-vm)**2)
        # 按照误差值排序，并去掉最大和最小的 5% 的值
        lower_percentile = np.percentile(errors, percentile)
        upper_percentile = np.percentile(errors, 100 - percentile)
        # 过滤掉小于 lower_percentile 或大于 upper_percentile 的值
        filtered_errors = errors[(errors >= lower_percentile) & (errors <= upper_percentile)]
        # 计算剩余数据的平均值
        aee = np.mean(filtered_errors)
    else:
        aee = np.mean(np.sqrt((ut-um)**2+(vt-vm)**2))
    return aee

def AAE(ut, vt, um, vm):
    """
    3. Average Angular Error
    Ref: https://doi.org/10.1007/s00348-020-03062-x
    """
    cos_theta = (ut*um+vt*vm)/np.sqrt((ut**2+vt**2)*(um**2+vm**2)+1e-16)
    cos_theta = np.clip(cos_theta, a_min=-1.0, a_max=1.0)
    theta = np.arccos(cos_theta)
    return np.mean(theta)
    
def SSIM(ut, vt, um, vm):
    """
    4. Structural Similarity
    Ref: https://doi.org/10.1017/jfm.2022.135
    """
    img1 = np.concatenate([ut, vt])
    img2 = np.concatenate([um, vm])
    
    # ssim_score = structural_similarity(img1, img2, multichannel=False) # channel_axis
    # ssim_score = structural_similarity(img1, img2, channel_axis=False) # channel_axis data_range=img.max() - img.min()
    ssim_score = structural_similarity(img1, img2, channel_axis=False, data_range=img1.max() - img1.min()) # channel_axis 
    return ssim_score

def ModC(ut, vt, um, vm):
    """
    5. Modulation coefficient
    Ref: our PIV-DCNN, https://doi.org/10.1007/s00348-017-2456-1 
    """
    ux = np.concatenate([ut, vt])
    uy = np.concatenate([um, vm])

    mc = np.sum(ux*uy)/np.sum(ux*ux+1e-16)
    return mc



def EPE1(ut, vt, um, vm):
    epe = EPE(ut, vt, um, vm)
    return np.sum(epe>1.0)

def EPE2(ut, vt, um, vm):
    epe = EPE(ut, vt, um, vm)
    return np.sum(epe>2.0)

def EPE3(ut, vt, um, vm):
    epe = EPE(ut, vt, um, vm)
    return np.sum(epe>3.0)

def PIVmetric(ut, vt, um, vm, show=True, percentile=None):
    rmse = RMSE(ut, vt, um, vm, percentile)
    aee = AEE(ut, vt, um, vm, percentile)
    aae = AAE(ut, vt, um, vm)
    ssim = SSIM(ut, vt, um, vm)
    mc = ModC(ut, vt, um, vm)
    epe1 = EPE1(ut, vt, um, vm)
    epe2 = EPE2(ut, vt, um, vm)
    epe3 = EPE3(ut, vt, um, vm)

    if show:
        r_str = f"{rmse:.4f}(RMSE)\t{aee:.4f}(AEE)\t{aae:.4f}(AAE,rad)\t{ssim:.4f}(SSIM)\t{mc:.4f}(MoC)\t{epe1}(epe1)\t{epe2}(epe2)\t{epe3}(epe3)\t"
        print(r_str)
    return rmse, aee, aae, ssim, mc, epe1, epe2, epe3


def test():
    um = 10+0.1*np.random.randn(512,512)
    vm = 5.+0.1*np.random.randn(512,512)
    
    ut = 10+0.0*np.random.randn(512,512)
    vt = 5.+0.0*np.random.randn(512,512)
    
    res = PIVmetric(ut,vt,um,vm)
    print(res)

if __name__ == "__main__":
    test()