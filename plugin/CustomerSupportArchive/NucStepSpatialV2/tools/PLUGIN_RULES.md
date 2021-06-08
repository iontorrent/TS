# Guidelines for plugin development alongside tools

## Rules
- Changes to metric definition require a major revision change for your plugin (major.minor.bug)
- At a minimum, define such major changes for posterity in a README.md file in your plugin

## Updates
- Update to use PluginMixin (then call self.init_plugin())
- Remove block_reshape (and analogs) definitions and leverage block_reshape.py
- Leverage lanes.py for iteration through lanes and other handy multilane tools
- Remove dependencies on average.py, as it is deprecated.