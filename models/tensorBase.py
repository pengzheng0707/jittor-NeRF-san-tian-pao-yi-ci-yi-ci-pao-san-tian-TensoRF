import jittor as jt
from jittor import init
from jittor import nn
from .sh import eval_sh_bases
import numpy as np
import time
from dataLoader.ray_utils import sample_pdf
jt.flags.use_cuda = 1

def positional_encoding(positions, freqs):
    
        freq_bands = (2**jt.arange(freqs).float()) # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = jt.concat([jt.sin(pts), jt.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - jt.exp(-sigma*dist)

    T = jt.cumprod(jt.concat([jt.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = nn.relu(jt.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(nn.Module):
    def __init__(self, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()


        self.aabb=aabb
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = jt.int64([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]])

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled).view(1,-1,1,1,3)    #TODO:
        if xyz_sampled.shape[1] == 0:
            return jt.zeros([0])
        else:
            return nn.grid_sample(self.alpha_volume, xyz_sampled, align_corners=True).view(-1)


    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape

        self.mlp = nn.Sequential(nn.Linear(self.in_mlpC, featureC), nn.ReLU(), nn.Linear(featureC, featureC), nn.ReLU(), nn.Linear(featureC,3))
        nn.init.constant_(self.mlp[-1].bias, 0)

    def execute(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = jt.concat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = jt.sigmoid(rgb)

        return rgb

class MLPRender_PE(nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe

        self.mlp = nn.Sequential(nn.Linear(self.in_mlpC, featureC), nn.ReLU(), nn.Linear(featureC, featureC), nn.ReLU(), nn.Linear(featureC,3))
        nn.init.constant_(self.mlp[-1].bias, 0)

    def execute(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = jt.concat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = jt.sigmoid(rgb)

        return rgb

class MLPRender(nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe

        self.mlp = nn.Sequential(nn.Linear(self.in_mlpC, featureC), nn.ReLU(), nn.Linear(featureC, featureC), nn.ReLU(), layer3 = nn.Linear(featureC,3))
        nn.init.constant_(self.mlp[-1].bias, 0)

    def execute(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = jt.concat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = jt.sigmoid(rgb)

        return rgb



class TensorBase(nn.Module):
    def __init__(self, aabb, gridSize, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask


        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio


        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]


        self.init_svd_volume(gridSize[0])

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    @jt.no_grad()   #TODO:
    def update_stepSize(self, gridSize):
        self.aabb=jt.array(self.aabb)
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= jt.int64(gridSize)
        self.units=(self.aabbSize / (self.gridSize-1)).float32()
        self.stepSize=(jt.mean(self.units)*self.step_ratio)
        self.aabbDiag = jt.sqrt(jt.sum(jt.sqr(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb})
        jt.save(ckpt, path)


    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = jt.float64(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask( ckpt['alphaMask.aabb'], alpha_volume.float())
            self.alphaMask.aabb=jt.array(self.alphaMask.aabb)
            self.alphaMask.aabbSize=jt.array(self.alphaMask.aabbSize)
            self.alphaMask.invgridSize=jt.array(self.alphaMask.invgridSize) #TODO:
        self.load_state_dict(ckpt['state_dict'])
        para=[]
        for a in ['density_','app_']:
            for b in ['line.','plane.']:
                for c in ['0','1','2']:
                    para.append(a+b+c)
        for i in range(3):
            self.density_line[i]=ckpt['state_dict'][para[i]]
        for i in range(3):
            self.density_plane[i]=ckpt['state_dict'][para[i+3]]
        for i in range(3):
            self.app_line[i]=ckpt['state_dict'][para[i+6]]
        for i in range(3):
            self.app_plane[i]=ckpt['state_dict'][para[i+9]]

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = jt.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += jt.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, jt.logical_not(mask_outbbox)

    def sample_ray(self, rays_o, rays_d, dx, is_train=True, N_samples=-1):
        N_samples = N_samples+1 if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = jt.where(rays_d==0, jt.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = jt.minimum(rate_a, rate_b).max(-1).clamp(near, far)

        rng = jt.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += jt.rand_like(rng[:,[0]])
        step = stepsize * rng
        interpx = (t_min[...,None] + step)
        # interpx = jt.linspace(0., 1., N_samples)
        # interpx = near + (far - near) * interpx
        # interpx = jt.broadcast(interpx, [rays_o.shape[0], N_samples])
        means, covs = self.cast_rays(interpx, rays_o, rays_d, dx)
        rays_pts=means+covs*jt.randn(covs.shape)
        #rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, jt.logical_not(mask_outbbox)

    def cast_rays(self,t_samples, origins, directions, radii, diagonal=True):
        """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
        Args:
            t_samples: float array [B, n_sample+1], the "fencepost" distances along the ray.
            origins: float array [B, 3], the ray origin coordinates.
            directions [B, 3]: float array, the ray direction vectors.
            radii[B, 1]: float array, the radii (base radii for cones) of the rays.
            ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
            diagonal: boolean, whether or not the covariance matrices should be diagonal.
        Returns:
            a tuple of arrays of means and covariances.
        """
        t0 = t_samples[..., :-1]  # [B, n_samples]
        t1 = t_samples[..., 1:]

        gaussian_fn = self.conical_frustum_to_gaussian

        means, covs = gaussian_fn(directions, t0, t1, radii, diagonal)
        means = means + jt.unsqueeze(origins, dim=-2)
        return means, covs

    def conical_frustum_to_gaussian(self, directions, t0, t1, base_radius, diagonal, stable=True):
        """Approximate a conical frustum as a Gaussian distribution (mean+cov).
        Assumes the ray is originating from the origin, and base_radius is the
        radius at dist=1. Doesn't assume `directions` is normalized.
        Args:
            directions: torch.tensor float32 3-vector, the axis of the cone
            t0: float, the starting distance of the frustum.
            t1: float, the ending distance of the frustum.
            base_radius: float, the scale of the radius as a function of distance.
            diagonal: boolean, whether or the Gaussian will be diagonal or full-covariance.
            stable: boolean, whether or not to use the stable computation described in
            the paper (setting this to False will cause catastrophic failure).
        Returns:
            a Gaussian (mean and covariance).
        """

        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
        t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                            (3 * mu ** 2 + hw ** 2) ** 2)
        r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                    (hw ** 4) / (3 * mu ** 2 + hw ** 2))

        return self.lift_gaussian(directions, t_mean, t_var, r_var)

    def lift_gaussian(self,directions, t_mean, t_var, r_var):
        """Lift a Gaussian defined along a ray to 3D coordinates."""
        mean = jt.unsqueeze(directions, dim=-2) * jt.unsqueeze(t_mean, dim=-1)  # [B, 1, 3]*[B, N, 1] = [B, N, 3]
        d_norm_denominator = jt.sum(directions ** 2, dim=-1, keepdims=True) + 1e-10
        # min_denominator = jt.full_like(d_norm_denominator, 1e-10)
        # d_norm_denominator = jt.maximum(min_denominator, d_norm_denominator)

        d_outer_diag = directions ** 2  # eq (16)
        null_outer_diag = 1 - d_outer_diag / d_norm_denominator
        t_cov_diag = jt.unsqueeze(t_var, dim=-1) * jt.unsqueeze(d_outer_diag,
                                                                    dim=-2)  # [B, N, 1] * [B, 1, 3] = [B, N, 3]
        xy_cov_diag = jt.unsqueeze(r_var, dim=-1) * jt.unsqueeze(null_outer_diag, dim=-2)
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag

    def shrink(self, new_aabb, voxel_size):
        pass

    @jt.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = jt.stack(jt.meshgrid(
            jt.linspace(0, 1, gridSize[0]),
            jt.linspace(0, 1, gridSize[1]),
            jt.linspace(0, 1, gridSize[2]),
        ), -1)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = jt.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((int(gridSize[1]), int(gridSize[2])))
        t=alpha.max(-1)
        t2=t.max(-1)
        return alpha, dense_xyz

    @jt.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2)
        alpha = alpha.clamp(0,1).transpose(0,2)[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = nn.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.min(0)
        xyz_max = valid_xyz.max(0)

        new_aabb = jt.stack((xyz_min, xyz_max))

        total = jt.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @jt.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()
        a=all_rays.shape[:-1]
        b=list(a)
        c=np.array(b)
        d=c.prod()
        N = jt.int64(d)

        mask_filtered = []
        idx_chunks = jt.split(jt.arange(N.item()), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk]

            rays_o, rays_d, dx  = rays_chunk[..., :3], rays_chunk[..., 3:6], rays_chunk[..., 6:]
            if bbox_only:
                vec = jt.where(rays_d == 0, jt.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = jt.minimum(rate_a, rate_b).max(-1)#.clamp(near, far)
                t_max = jt.maximum(rate_a, rate_b).min(-1)#.clamp(near, far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, dx, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox)

        mask_filtered = jt.concat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {jt.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return nn.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return nn.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = jt.ones_like(xyz_locs[:,0]).bool()
            

        sigma = jt.zeros(xyz_locs.shape[:-1])

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - jt.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha


    def execute(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        N_importance=0
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        dx=rays_chunk[:, 6:]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = jt.concat((z_vals[:, 1:] - z_vals[:, :-1], jt.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = jt.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, dx, is_train=is_train,N_samples=N_samples)
            dists = (z_vals[:, 1:] - z_vals[:, :-1]) * jt.norm(jt.unsqueeze(viewdirs, dim=-2), dim=-1)
            z_vals = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = jt.logical_not(ray_valid)
            ray_invalid[ray_valid] |= (jt.logical_not(alpha_mask))
            ray_valid = jt.logical_not(ray_invalid)


        sigma = jt.zeros(xyz_sampled.shape[:-1])
        rgb = jt.zeros((*xyz_sampled.shape[:2], 3))

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)


        app_mask = weight > self.rayMarch_weight_thres
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = jt.sum(weight, -1)
        rgb_map = jt.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and jt.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clamp(0,1)

        with jt.no_grad():
            depth_map = jt.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

