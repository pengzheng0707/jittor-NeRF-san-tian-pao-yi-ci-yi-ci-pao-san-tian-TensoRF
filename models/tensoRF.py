from .tensorBase import *
jt.flags.use_cuda = 1

class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, **kargs):
        super(TensorVM, self).__init__(aabb, gridSize,  **kargs)
        

    def init_svd_volume(self, res):
        self.plane_coef = 0.1 * jt.randn((3, self.app_n_comp + self.density_n_comp, res, res))
        self.line_coef = 0.1 * jt.randn((3, self.app_n_comp + self.density_n_comp, res, 1))
        self.basis_mat = nn.Linear(self.app_n_comp * 3, self.app_dim, bias=False)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_features(self, xyz_sampled):

        coordinate_plane = jt.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach()
        coordinate_line = jt.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = jt.stack((jt.zeros_like(coordinate_line), coordinate_line), dim=-1).detach()

        plane_feats = nn.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = nn.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = jt.sum(plane_feats * line_feats, dim=0)
        
        
        plane_feats = nn.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = nn.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        app_features = self.basis_mat((plane_feats * line_feats).transpose([1,0]))
        
        return sigma_feature, app_features

    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = jt.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = jt.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = jt.stack((jt.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_feats = nn.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = nn.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = jt.sum(plane_feats * line_feats, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = jt.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = jt.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = jt.stack((jt.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = nn.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = nn.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).transpose([1,0]))
        
        
        return app_features
    

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            # print(self.line_coef.shape, vector_comps[idx].shape)
            n_comp, n_size = vector_comps[idx].shape[:-1]
            
            dotp = jt.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape)
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape,non_diagonal.shape)
            total = total + jt.mean(jt.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        
        return self.vectorDiffs(self.line_coef[:,-self.density_n_comp:]) + self.vectorDiffs(self.line_coef[:,:self.app_n_comp])
    
    
    #TODO:@jt.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef[i] = nn.interpolate(plane_coef[i].detach(), size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear', align_corners=True)
            line_coef[i] = nn.interpolate(line_coef[i].detach(), size=(res_target[vec_id], 1), mode='bilinear', align_corners=True)

        # plane_coef[0] = nn.Parameter(
        #     nn.interpolate(plane_coef[0].data, size=(res_target[1], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[0] = nn.Parameter(
        #     nn.interpolate(line_coef[0].data, size=(res_target[2], 1), mode='bilinear', align_corners=True))
        # plane_coef[1] = nn.Parameter(
        #     nn.interpolate(plane_coef[1].data, size=(res_target[2], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[1] = nn.Parameter(
        #     nn.interpolate(line_coef[1].data, size=(res_target[1], 1), mode='bilinear', align_corners=True))
        # plane_coef[2] = nn.Parameter(
        #     nn.interpolate(plane_coef[2].data, size=(res_target[2], res_target[1]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[2] = nn.Parameter(
        #     nn.interpolate(line_coef[2].data, size=(res_target[0], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    #TODO:@jt.no_grad()
    def upsample_volume_grid(self, res_target):
        # self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        # self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        with jt.no_grad():
            scale = res_target[0]/self.line_coef.shape[2] #assuming xyz have the same scale
            plane_coef = nn.interpolate(self.plane_coef.detach(), scale_factor=scale, mode='bilinear',align_corners=True)
            line_coef  = nn.interpolate(self.line_coef.detach(), size=(res_target[0],1), mode='bilinear',align_corners=True)
        self.plane_coef, self.line_coef = plane_coef, line_coef

        self.compute_stepSize(res_target)
        print(f'upsamping to {res_target}')


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, **kargs)


    def init_svd_volume(self, res):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1)
        self.basis_mat = nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False)


    def init_one_svd(self, n_component, gridSize, scale):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(
                scale * jt.randn((1, n_component[i], int(gridSize[mat_id_1]), int(gridSize[mat_id_0]))))  #
            line_coef.append(scale * jt.randn((1, n_component[i], int(gridSize[vec_id]), 1)))

        return nn.ParameterList(plane_coef), nn.ParameterList(line_coef)
    
    

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = jt.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + jt.mean(jt.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + jt.mean(jt.abs(self.density_plane[idx])) + jt.mean(jt.abs(self.density_line[idx]))# + jt.mean(jt.abs(self.app_plane[idx])) + jt.mean(jt.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 #+ reg(self.app_line[idx]) * 1e-3
        return total

    def compute_densityfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = jt.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = jt.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = jt.stack((jt.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = jt.zeros((xyz_sampled.shape[0],))
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = nn.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = nn.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + jt.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature


    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = jt.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = jt.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = jt.stack((jt.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(nn.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(nn.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = jt.concat(plane_coef_point), jt.concat(line_coef_point)

 
        return self.basis_mat((plane_coef_point * line_coef_point).transpose([1,0]))



    #TODO:@jt.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):

            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = nn.interpolate(plane_coef[i].detach(), size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear', align_corners=True)  #TODO:
            line_coef[i] = nn.interpolate(line_coef[i].detach(), size=(res_target[vec_id], 1), mode='bilinear', align_corners=True)


        return plane_coef, line_coef

    #TODO:@jt.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    #TODO:
    def shrink(self, new_aabb):
        with jt.no_grad():
            print("====> shrinking ...")
            xyz_min, xyz_max = new_aabb
            t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
            # print(new_aabb, self.aabb)
            # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
            t_l, b_r = jt.round(jt.round(t_l)).int64(), jt.round(b_r).int64() + 1
            b_r = jt.stack([b_r, self.gridSize]).min(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = self.density_line[i].detach()[...,t_l[mode0].item():b_r[mode0].item(),:]
            self.app_line[i] = self.app_line[i].detach()[...,t_l[mode0].item():b_r[mode0].item(),:]
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = self.density_plane[i].detach()[...,t_l[mode1].item():b_r[mode1].item(),t_l[mode0].item():b_r[mode0].item()]
            self.app_plane[i] = self.app_plane[i].detach()[...,t_l[mode1].item():b_r[mode1].item(),t_l[mode0].item():b_r[mode0].item()]

        with jt.no_grad():
            if not jt.all(self.alphaMask.gridSize == self.gridSize):
                t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
                correct_aabb = jt.zeros_like(new_aabb)
                correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
                correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
                print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
                new_aabb = correct_aabb

            newSize = b_r - t_l
            self.aabb = new_aabb
            self.update_stepSize(newSize)


class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, **kargs):
        super(TensorCP, self).__init__(aabb, gridSize, **kargs)


    def init_svd_volume(self, res):
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2)
        self.app_line = self.init_one_svd(self.app_n_comp[0], self.gridSize, 0.2)
        self.basis_mat = nn.Linear(self.app_n_comp[0], self.app_dim, bias=False)


    def init_one_svd(self, n_component, gridSize, scale):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append( scale * jt.randn((1, n_component, gridSize[vec_id], 1)))
        return nn.ParameterList(line_coef)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):

        coordinate_line = jt.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = jt.stack((jt.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = nn.grid_sample(self.density_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * nn.grid_sample(self.density_line[1], coordinate_line[[1]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * nn.grid_sample(self.density_line[2], coordinate_line[[2]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        sigma_feature = jt.sum(line_coef_point, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):

        coordinate_line = jt.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = jt.stack((jt.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = nn.grid_sample(self.app_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * nn.grid_sample(self.app_line[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * nn.grid_sample(self.app_line[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.transpose([1,0]))
    

    #TODO:@jt.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = nn.interpolate(density_line_coef[i].detach(), size=(res_target[vec_id], 1), mode='bilinear', align_corners=True)
            app_line_coef[i] = nn.interpolate(app_line_coef[i].detach(), size=(res_target[vec_id], 1), mode='bilinear', align_corners=True)

        return density_line_coef, app_line_coef

    @jt.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    #TODO:@jt.no_grad()
    def shrink(self, new_aabb):
        with jt.no_grad():
            print("====> shrinking ...")
            xyz_min, xyz_max = new_aabb
            t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

            t_l, b_r = jt.round(jt.round(t_l)).int64(), jt.round(b_r).int64() + 1
            b_r = jt.stack([b_r, self.gridSize]).min(0)


        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = self.density_line[i].detach()[...,t_l[mode0]:b_r[mode0],:]
            self.app_line[i] = self.app_line[i].detach()[...,t_l[mode0]:b_r[mode0],:]
        with jt.no_grad():
            if not jt.all(self.alphaMask.gridSize == self.gridSize):
                t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
                correct_aabb = jt.zeros_like(new_aabb)
                correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
                correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
                print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
                new_aabb = correct_aabb

            newSize = b_r - t_l
            self.aabb = new_aabb
            self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + jt.mean(jt.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total