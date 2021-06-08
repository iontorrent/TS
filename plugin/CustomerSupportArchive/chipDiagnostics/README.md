# Info
- This readme file is meant to contain update information to images and metrics that go along with the plugin.
- **At a minimum, major version changes must be made when metric definitions change, and they should be described in this file!**
- Further detail can be handy to retain.  Previously, some of this recent information was included in the docstrings for plugin classes.

---

# Major Revisions

## **6.0.0** [Feb 2019]
- After realization that 560 chips and others did not adhere to the new Valkyrie rules and confirmation that the 'DynamicRange' field is non-deterministic and saves the DR at save time into this field.....so not useful.
- Therefore, reverted to DynamicRangeAfterBF.

## **5.0.X** [Jan 2019]
- Thought it was time to switch back to DynamicRange as the true metric.  Did not turn out to be the right decision.

## **4.0.0** [Sept 2019]
- The primary change here redefined our pixel offset metric definitions based on how we have been pulling in the wrong DR for some time now.  The new value is correctly pulled from the explog file (DynamicRangeAfterBF).
- Previous versions of the plugin still have this information (so we can retroactively correct on ChipDB) so that they can be corrected by the factor of ( DynamicRangeAfterBF / DynamicRange )
- Also created a host of new metrics (usually block-reshape based ones) with a new prefix of 'true_' which are now appropriately calculated due to exclusion of pinned pixels from calculations.  (This is particularly important for studies of the square defect on GX5/GX7 chips)

## **3.0.0** [June 2018]
- Added (radial) edge analysis capabilities for calibration metrics.

# Other Revisions of Note

## 3.3.0

- This was the first revision built to work on either RUO TS or the Valkyrie TS.