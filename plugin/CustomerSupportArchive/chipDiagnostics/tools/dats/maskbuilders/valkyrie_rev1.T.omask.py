import matplotlib.pyplot as plt
from tools import chipcal
import numpy as np
from tools import chiptype
from tools import chip

# Lane 1: http://ecc.itw/results/540/QPG392/02/X4_Y6_TPE_2017_06_29_14_08/ecc_summary.html
# Lane 2: http://ecc.itw/results/540/QPG392/02/X4_Y6_TPE_2017_06_29_14_45/ecc_summary.html
# Lane 3: http://ecc.itw/results/540/QPG392/02/X4_Y6_TPE_2017_06_29_13_26/ecc_summary.html
# Lane 4: http://ecc.itw/results/540/QPG392/02/X4_Y6_TPE_2017_06_29_12_31/ecc_summary.html

if False:
    dirnames = [ 'lane%s' % s for s in range( 1, 5 ) ]

    gain = None
    for dn in dirnames:
        cc = chipcal.ChipCal( dn, chiptype='541' )
        cc.load_gain()
        if gain is None:
            gain = cc.gain
        else:
            gain = np.max( [ gain, cc.gain ], axis=0 )

    mask = gain > 500
    print 'Mask Size: %s' % ( str( mask.shape ) )
    mask.tofile( 'activemask.dat' )

ct = chiptype.ChipType( '541' )
mask = np.fromfile( 'activemask.dat', dtype=np.bool ).reshape( ct.chipR, ct.chipC )
extent = ( 0, mask.shape[1], 0, mask.shape[0] )

plt.figure()
plt.imshow( mask[::10,::10], extent=extent, origin='lower' )

plt.figure()
rowavg = mask.mean( axis=1 )
plt.plot( rowavg )
colavg = mask.mean( axis=0 )
colavgmask = colavg > 0.8
plt.plot( colavg )

mask[:,~colavgmask] = 0
plt.figure()
plt.imshow( mask[::10,::10], extent=extent, origin='lower' )


colavg = mask[2000:-2000,:].mean( axis=0 )
colavgmask = colavg > 0.5
mask[:,~colavgmask] = 0
rowavgmask = rowavg > 0.8
#print mask.sum()
#print mask[rowavgmask,:][:,colavgmask].sum()
#print mask[rowavgmask,:][:,colavgmask].size
#mask[rowavgmask,:][:,colavgmask] = 1
mask[ np.logical_and( *np.meshgrid( colavgmask, rowavgmask ) ) ] = 1
plt.imshow( mask[::10,::10], extent=extent, origin='lower' )


filename = 'valkyrie_rev1.T.omask'
chip.make_mask( mask, filename )
mask = chip.load_mask( filename, 'Valkyrie_3528' )
plt.figure()
#plt.imshow( mask[::10,::10], extent=extent, origin='lower' )
plt.imshow( mask[::10,::10], origin='lower' )
