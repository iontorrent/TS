FieldSupport
===========

**Do not modify plugins inside the rndplugins directory!**
Plugins should be modified upsteam and then pulled into this repo with the `fab update_plugins` command. 
Plugins can check for the existence of the `TSP_LIMIT_OUTPUT` environment variable if needed. This variable is only set
by the FieldSupport runtime. Plugins may need to run in a limited output mode inside this plugin to keep
the size of the resulting archive small (<10MB).

Updating Plugins
-----------------
Command: `fab update_plugins`

You will need the python fabric library installed. See [http://www.fabfile.org/](fabfile.org)

