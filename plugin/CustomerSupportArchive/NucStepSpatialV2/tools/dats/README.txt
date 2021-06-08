File decriptions
# Due to the development history, it is not always clear what each file is.
# Users are requested to load and verify each mask.
# Once a mask is verified, removed the "verified: NO" line from the mask description

521.omask
  Author:      Yating
  Date:        <6/2017
  Description: 521 mask.  It's unclear if this is gluesafe or not
  prev:        521mask.dat
  verified:    NO

521_gluesafe.omask
  Author:      Yating
  Date:        <6/2017
  Description: 521 mask. Pixels are guaranteed to be inside the glueline
  prev:        521mask_small.dat
  verified:    NO

550_16.omask
  Author:      Yating
  Date:        <6/2017
  Description: Defines the edge of the reference pixels for the 550_16 chiptype
  verified:    NO

550.omask
  Description: symlink to 550_3491.omask. Preserved for backwards compatibility

550_3490.omask
  Author:      Scott
  Date:        9/13/2017
  Description: Defines edge of reference pixels for 550_3490 chiptype. Cropped from 550_3491

550_3491.omask
  Author:      Yating
  Date:        <6/2017
  Description: Defines edge of reference pixels for 550_3491 chiptype

550_3525.omask
  Author:      Scott
  Date:        9/13/2017
  Description: Defines edge of reference pixels for 550_3425 chiptype. Cropped from 550_3491

p1mask.dat
  Author:      Scott
  Date:        <6/2017
  Description: Early version of an omask file, developed specifically for P1v3 in tools/sequencing
               Does not contain chip omask header

p1.omask
  Author:      Yating
  Date:        <6/2017
  Description: 540 mask. Pixels are nominally inside the glueline
  prev:        p1mask0.dat
  verified:    NO

p1_v2.omask
  Author:      Yating
  Date:        7/2017
  Description: Updated p1.omask. Pixels are nominally inside the glueline
  prev:        p1.omask

spa_mask.omask
  Author:      Yating
  Date:        8/2017
  Description: spa_tn size of mask based on 530 chip flowcell

p1_gluesafe.omask
  Author:      Yating
  Date:        <6/2017
  Description: 540 mask. Pixels are guaranteed to be inside the glueline
  prev:        p1mask00.dat
  verified:    NO

p1_gluesafe_v2.omask
  Author:      Yating
  Date:        7/2017
  Description: Upated p1_gluesafe.omask
  prev:        p1_gluesafe.omask

valkyrie_rev1.T.omask
  Author:      Scott
  Date:        7/24/207
  Description: First attempt at 4-lane Valkyrie chip.  Very primitive.  Pixels are nominally
               inside the glue lines.  Mask is transposed due to lane transposition

vfc1.txt
  Author:      Datacollect
  Date:        <6/2017
  Description: VFC compression profile

vfc2.txt
  Author:      Datacollect
  Date:        <6/2017
  Description: VFC compression profile for 550 chips

vfc3.txt
  Author:      Datatollect since 3526
  Date:        <6/2017
  Description: VFC compression profile for 550 chips since version 3526

vfc4.txt
  Author:      Datatollect since 3526
  Date:        7/25/2017
  Description: Uncompressed VFC profile 
