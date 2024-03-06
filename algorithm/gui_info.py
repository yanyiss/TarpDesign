class GUI_INFO():
    def __init__(self,deform):
        self.nv=deform.tarp.vertices.shape[0]
        self.nf=deform.tarp.faces.shape[0]
        self.vertices=self.transfer(deform.tarp.vertices[0])
        self.faces=self.transfer(deform.tarp.faces[0])
        self.forces=self.transfer(deform.force.now_force()[0])
        self.boundary_index=self.transfer(deform.force.boundary_index)

    def transfer(self,tensor):
        return tensor.clone().detach().cpu().numpy()
    
    def update(self,deform):
        self.vertices=self.transfer(deform.tarp.vertices[0])
        self.forces=self.transfer(deform.force.now_force()[0])