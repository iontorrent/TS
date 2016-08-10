# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved
from __future__ import division
import tables
import numpy as np
import os
import os.path


class DebugParams(object):
    def __init__(self,path_to_sigproc):
        self.__path = path_to_sigproc
        self.__nuc_param = {'sigma':'derived_param','midNucTime':'derived_param','krate':'enzymatics','NucModifyRatio':'buffering', 'd':'enzymatics', 't_mid_nuc_delay':'nuc_shape', 'Concentration':'nuc_shape', 'sigma_mult':'nuc_shape', 'kmax':'enzymatics'}
        self.__flow_param = {'t_mid_nuc':'nuc_shape', 't_mid_nuc_shift_per_flow':'nuc_shape','darkness':'misc'}
        self.__flow_group_param = {'tau_R_o':'buffering', 'tau_R_m':'buffering', 'tauE':'buffering', 'RatioDrift':'buffering', 'valve_open':'nuc_shape', 'magic_divisor_for_timing':'nuc_shape','CopyDrift':'misc', 'molecules_to_micromolar':'misc', 'tshift':'misc', 'sens':'misc', 'SENSMULTIPLIER':'misc' } #'sigma':'nuc_shape', 
        self.__nuc_map={"T":0, "A":1, "C":2, "G":3}

    def getRegionIdxByPos( self, pos ):
        reg_corner = np.floor_divide( pos, self.step )*self.step
        ind = np.where((self.loc==reg_corner).all(axis=1))
        return (ind[0] if ind[0].size>0 else -1).min()

    def __get_nuc_attr( self, name, attr, flow, nuc, reg ):
        flow_group = int(flow // self.nFlowsPerGroup)
        nuc_idx = self.__nuc_map[nuc]
        full_name = name+'_'+str(nuc_idx)
        full_attr= ('/region/region_param/'+attr) if attr!='derived_param' else '/region/derived_param'
        namedParameterList=np.array(self.__region_param.getNodeAttr(full_attr,'paramNames').split(','))
        param_idx=np.where(namedParameterList==full_name)[0][0]
        val = self.__region_param.getNode(full_attr)
        return val[reg,param_idx,flow_group]

    def __get_flow_group_attr( self, name, attr, flow, nuc, reg ):
        flow_group = int(flow // self.nFlowsPerGroup)
        full_name = name
        full_attr='/region/region_param/'+attr
        namedParameterList=np.array(self.__region_param.getNodeAttr(full_attr,'paramNames').split(','))
        param_idx=np.where(namedParameterList==full_name)[0][0]
        val = self.__region_param.getNode(full_attr)
        return val[reg,param_idx,flow_group]

    def __get_flow_attr( self, name, attr, flow, nuc, reg ):
        flow_group = int(flow // self.nFlowsPerGroup)
        flow_idx = int(flow % self.nFlowsPerGroup)
        full_name = name+'_'+str(flow_idx)
        full_attr='/region/region_param/'+attr
        namedParameterList=np.array(self.__region_param.getNodeAttr(full_attr,'paramNames').split(','))
        param_idx=np.where(namedParameterList==full_name)[0][0]
        val = self.__region_param.getNode(full_attr)
        return val[reg,param_idx,flow_group]

    def __get_darkness_attr( self, reg, nuc ):
        full_attr='/region/darkMatter/missingMass'
        val = self.__region_param.getNode(full_attr).read()
        nuc_idx = self.__nuc_map[nuc]
        return val[reg,nuc_idx,:]

    def getRegionParam( self, reg, name, flow, nuc ):
        try:
            attr = self.__nuc_param[name]
            return( self.__get_nuc_attr( name, attr, flow, nuc, reg ) )
        except: pass
        try:
            attr = self.__flow_param[name]
            return( self.__get_flow_attr( name, attr, flow, nuc, reg ) )
        except: pass
        try:
            attr = self.__flow_group_param[name]
            return( self.__get_flow_group_attr( name, attr, flow, nuc, reg ) )
        except:
            return

    def getBgRegionParams( self, pos, flow, nuc ):
        ret = dict()
        reg = tuple( self.getRegionIdxByPos(pos_1) for pos_1 in pos )
        for (par, temp) in self.__nuc_param.items()+self.__flow_param.items()+self.__flow_group_param.items():
            ret[par]=self.getRegionParam(reg, par, flow, nuc)
        ret['missingMass']=tuple(self.__get_darkness_attr( r, nuc ) for r in reg)
        return(ret)

    def _getRegionParams(self,pos):
        attributes=('/region/region_param/nuc_shape','/region/region_param/misc','/region/region_param/enzymatics','/region/region_param/buffering')
        params={}

        for attr in attributes:
            namedParameterList=self.__region_param.getNodeAttr(attr,'paramNames').split(',')
            namedParameterList = namedParameterList[:-1] if namedParameterList[-1]=='' else namedParameterList
            param_values = self.__region_param.getNode(attr)

            for k in range(len(namedParameterList)):
                val = [ param_values[ self.getRegionIdxByPos(well_pos),k,:] for well_pos in pos ]
                params[namedParameterList[k]]=val[0]

        return params

    def getBeadParams(self,pos,flow=slice(None)):
        params={}
        params['kmult']=[self.__data['kmult'][row,col,flow] for row,col in pos]
        params['copies']=[self.__data['bead_base_parameters'][row,col,0] for row,col in pos]
        params['etbR']=[self.__data['bead_base_parameters'][row,col,1] for row,col in pos]
        params['dmult']=[self.__data['bead_base_parameters'][row,col,2] for row,col in pos]
        params['gain']=[self.__data['bead_base_parameters'][row,col,3] for row,col in pos]
        params['deltaTime']=[self.__data['bead_base_parameters'][row,col,4] for row,col in pos]
        params['amplitude']=[self.__data['amplitude'][row,col,flow] for row,col in pos]
        params['resError']=[self.__data['residual_error'][row,col,flow] for row,col in pos]
        #params['errByBlock']=[self.__data['average_error_by_block'][row,col,flow] for row,col in pos]
        params['dcOffset']=[self.__data['trace_dc_offset'][row,col,flow] for row,col in pos]
        return params

    def _getRegParams( self, pos ):
        params=self._getRegionParams(pos)
        return params

    def LoadData(self):
        self.__region_param = tables.openFile(os.path.join(self.__path,'region_param.h5'))
        self.__bead_param = tables.openFile(os.path.join(self.__path,'bead_param.h5'))
        self.__data={}
        self.__data['kmult'] = self.__bead_param.getNode('/bead/kmult')
        self.__data['bead_base_parameters'] = self.__bead_param.getNode('/bead/bead_base_parameters')
        self.__data['amplitude'] = self.__bead_param.getNode('/bead/amplitude')
        self.__data['residual_error'] = self.__bead_param.getNode('/bead/residual_error')
        self.__data['average_error_by_block'] = self.__bead_param.getNode('/bead/average_error_by_block')
        self.__data['trace_dc_offset'] = self.__bead_param.getNode('/bead/trace_dc_offset')

        self.nFlows = self.__data['amplitude'].shape[2]
        self.nFlowsPerGroup = int(self.nFlows / self.__region_param.getNode('/region/region_param/nuc_shape').shape[2])
        self.loc=np.squeeze(np.array(self.__region_param.getNode('/region/region_location')),axis=2)
        if self.loc.shape[0]>1:
            cols=self.loc[:,0]
            rows=self.loc[:,1]
            self.step = np.array( [np.diff(np.sort(np.unique(cols)))[0], np.diff(np.sort(np.unique(rows)))[0]] )
        else:
            self.step=100000

    def getParamsForWells(self, pos):
        params = self.getBeadParams( pos )
        params1=self._getRegParams( pos )
        params.update(params1)
        return params

    def getEmptyTrace( self, pos, flow ):
        reg = self.getRegionIdxByPos(pos)
        return( self.__region_param.getNode('/region/empty_trace')[reg,:,flow] )


