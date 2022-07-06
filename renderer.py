import fractions
import os,imageio,sys
import jittor as jt
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False):

    rgbs, alphas, depth_maps, weights, uncertainties,ref_depth_maps, bk_depth_maps, trans_rgb_maps,frac_maps, ref_rgb_maps = [], [], [], [], [], [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk]
    
        rgb_map, depth_map, ref_depth_map, bk_depth_map, trans_rgb_map, frac_map, ref_rgb_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        ref_depth_maps.append(ref_depth_map)
        bk_depth_maps.append(bk_depth_map)
        trans_rgb_maps.append(trans_rgb_map)
        frac_maps.append(frac_map)
        ref_rgb_maps.append(ref_rgb_map)

    
    return jt.concat(rgbs), None, jt.concat(depth_maps), None, None, jt.concat(ref_depth_maps), jt.concat(bk_depth_maps), jt.concat(trans_rgb_maps), jt.concat(frac_maps), jt.concat(ref_rgb_maps)

@jt.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=False):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)
    os.makedirs(savePath+"/ref_dep", exist_ok=True)
    os.makedirs(savePath+"/bk_dep", exist_ok=True)
    os.makedirs(savePath+"/trans_rgb", exist_ok=True)
    os.makedirs(savePath+"/fra", exist_ok=True)
    os.makedirs(savePath+"/ref_rgb", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _, ref_dep_map, bk_dep_map, trans_rgb_map, fra_map, ref_rgb_map = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map, ref_dep_map, bk_dep_map, trans_rgb_map, fra_map ,ref_rgb_map= rgb_map.reshape(H, W, 3), depth_map.reshape(H, W), ref_dep_map.reshape(H,W), bk_dep_map.reshape(W,W), trans_rgb_map.reshape(H,W,3), fra_map.reshape(H,W), ref_rgb_map.reshape(H,W,3)

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        ref_dep_map, _ = visualize_depth_numpy(ref_dep_map.numpy(),near_far)
        bk_dep_map, _ = visualize_depth_numpy(bk_dep_map.numpy(),near_far)
        fra_map, _ = visualize_depth_numpy(fra_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = jt.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex')
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg')
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        trans_rgb_map = (trans_rgb_map.numpy() * 255).astype('uint8')
        ref_rgb_map = (ref_rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:d}.png', rgb_map)
            depth_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', depth_map)
            ref_dep_map = np.concatenate((rgb_map, ref_dep_map), axis=1)
            imageio.imwrite(f'{savePath}/ref_dep/{prtx}{idx:03d}.png', ref_dep_map)
            bk_dep_map = np.concatenate((rgb_map, bk_dep_map), axis=1)
            imageio.imwrite(f'{savePath}/bk_dep/{prtx}{idx:03d}.png', bk_dep_map)
            trans_rgb_map = np.concatenate((rgb_map, trans_rgb_map), axis=1)
            imageio.imwrite(f'{savePath}/trans_rgb/{prtx}{idx:03d}.png', trans_rgb_map)
            fra_map = np.concatenate((rgb_map, fra_map), axis=1)
            imageio.imwrite(f'{savePath}/fra/{prtx}{idx:03d}.png', fra_map)
            ref_rgb_map = np.concatenate((rgb_map, ref_rgb_map), axis=1)
            imageio.imwrite(f'{savePath}/ref_rgb/{prtx}{idx:03d}.png', ref_rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@jt.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=False):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = jt.float32(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = jt.concat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3), depth_map.reshape(H, W)

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

