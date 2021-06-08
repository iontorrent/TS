###########################################################
# Updating your tools to whatever is released in the repo #
###########################################################
# Let's say that somebody else updates the pointer on the submodule
# When you do a 
    git pull
# on the main repo, you will see something like:
#   modified: tools (new commits)
# Generally, this means that your version of tools is different/out of date
# To fix this, just do:
    git submodule update
# This will update your submodule repo to the defined copy

# Summary for updating the main repo:
    git pull
    git submodule update
# Note that when you do this, if you are on a branch in the submodule, it will
# switch you to a detached head

#############################################
# Updating tools to the latest and greatest #
#############################################
# Download the latest copy
    #DON'T DO THIS: git submodule update --remote --recursive
    # Do this instead:
        git submodule update --remote tools
    # -or-
        git fetch
        git merge origin/master
    # This is particularly important on rndplugins where multiple instances of submodule exist
    # and we only want to update one of them
# Internally, this will point the tools directory at a specific commit on the tools repo
# Push the updated tools reference back into the repo
    git add tools
    git commit -m "updating tools"

#############################################
# Making changes to the tools direcotry     #
#############################################
# Option 1:
#   First make sure you are on an active branch
      cd tools
      git fetch
      git checkout master
#   This will check out the master branch and you can edit tools locally, just like a standard repo
#   You should be able to do things like
      git commit -a -m "message"
      git pull
      git push origin master
#   IMPORTANT: If your project has dependencies on new updates to tools, pushing updates to the project
#     will not push updates to tools.  You need to push those changes separatly from within the tools 
#     subdirectory

# Option 2:
#   Edit the tools repo directly.  
#   Push changes to the tools repo
#   Download the latest and greatest to your project
#   This works well if you are making small changes or are confident in your changes

# Option 3:
#   Edith the tools directory in your project (remember these changes are not saved or tracked)
#   Test your code
#   Once you feel comfortable with your changes, move the changes into the tools repo using 
#     vimdiff or similar tools and push to the remote tools repo

###########################################################
# What to do if your tools directory has modified content #
###########################################################
# I don't know yet

###################################################################
# What to do if you want to roll back to a previous tools version #
###################################################################
    cd tools
# Download the history
    git fetch
# Checkout the target copy
    git checkout <SHA>
# Delete any files that you don't need

#######################################
# Adding tools to a new (git) project #
#######################################
# Go to your project directory
# Download the tools
    git submodule add ssh://git@stash.amer.thermo.com:7999/ecc/tools.git
# You should now also have a file .gitmodules
    git add .gitmodules tools
# Commit the repo
    git commit -m "adding tools"

######################################################
# Downloading a fresh copy of a repo that uses tools #
######################################################
# Clone the repo
    git clone <repo path>
# Fetch tools
    git submoduel update --init

###############################################################
# Updating existing local repo to new submodule configuration #
###############################################################
# Start on master branch
    git checkout master
# Delete the tools directory
    rm -r tools
# Fetch changes
    git pull
# Check the status
    git status
# If you see "deleted: tools", the restore it
    git reset --hard # WARNING: This will wipe away any uncommited changes in your repo
# Download the tools module
    git submodule update --init
# Note:  the tools directory is not "tracked".  Changes to tools are not 
#        perminant and will not be pushed to a repo (i.e. the head is disconnected)

# Troubleshooting: 
#   When all is said and done, you should NOT commit a "deleted: tools"
#   Hopefully git reset -- hard should resolve this
#   If, after merging from origin, things seem to be wrong, start with a dummy commit
     git commit -a -m "dummy commit"
#   IMPORTANT: DO NOT PUSH THIS COMIT TO ORIGIN
#   Roll back the last commit
      git reset HEAD~
#   This will undo the commit but leave all the incoming changes in place
#   Now delete all of the incoming changes
      git reset --hard
      git status
      rm -r <list all untracked files here>
#   You should once again have a clean copy
#   Try to download again
      git pull
      git status
#   This will probably leave you with a "delted: tools".  Reset again
      git reset --hard
      git status
#   The tools directory should now be present, but it should be empty.
#   Popultate it
      git submodule update --init

# More advanced troubleshooting
#   If you have existing changes (especially binary files like TRD.xlsx), it is best to do the following:
#     copy your binary files to a new location
        cp TRD.xlsx ~/TRD_tmp.xlsx
#     pull the remote files, but don't merge
        git fetch 
#     Copy the conflicting file into your branch
        git checkout origin/master -- TRD.xlsx
#     Commit the change
        git commit -a -m "Accepting remote TRD.xlsx"
#   Now return to start of section ("Updating existing...")
#   After sucessfull merge, take care of any additional manual merge of binary files and commit the changes
###########################################################
# Initial migration instructions (changing to submodule ) #
###########################################################
# After moving a directory (tools) to submodule on a branch, switch back to master
    git checkout -f master
# Merge in the branch
    git merge <branch>
# Delete the directory and restore it
    rm tools
    git reset --hard # WARNING: This will wipe away any uncommited changes in your repo
# Push changes to server
    git push origin master
