KENDO_TB = ./resources/less/kendo.tb.css
KENDO_TB_MIN = ./resources/less/kendo.tb.min.css
KENDO_TB_LESS = ./resources/less/kendo.tb.less
BOOTSTRAP = ./resources/bootstrap/css/bootstrap.css
BOOTSTRAP_MIN = ./resources/bootstrap/css/bootstrap.min.css
BOOTSTRAP_LESS = ./resources/bootstrap/less/bootstrap.less
BOOTSTRAP_RESPONSIVE = ./resources/bootstrap/css/bootstrap-responsive.css
BOOTSTRAP_RESPONSIVE_MIN = ./resources/bootstrap/css/bootstrap-responsive.min.css
BOOTSTRAP_RESPONSIVE_LESS = ./resources/bootstrap/less/responsive.less

TB_SYTLES_LESS = ./resources/less/tb-styles.less
TB_SYTLES = ./resources/styles/tb-styles.css
TB_SYTLES_MIN = ./resources/styles/tb-styles.min.css
TB_LAYOUT_LESS = ./resources/less/tb-layout.less
TB_LAYOUT = ./resources/styles/tb-layout.css
TB_LAYOUT_MIN = ./resources/styles/tb-layout.min.css
SIGNAL_LESS = ./resources/less/signal.less
SIGNAL = ./resources/styles/signal.css
SIGNAL_MIN = ./resources/styles/signal.min.css
IE7_LESS = ./resources/less/IE7.less
IE7_SYTLES = ./resources/styles/IE7.css
IE7_MIN = ./resources/styles/IE7.min.css
PRINT_LESS = ./resources/less/print.less
PRINT_SYTLES = ./resources/styles/print.css
PRINT_MIN = ./resources/styles/print.min.css

DATE=$(shell date +%I:%M%p)
CHECK=\033[32m✔\033[39m
HR=\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
hr=--------------------------------------------------

#
# BUILD DOCS
#
all: |compile-css #minify-js

compile-css:
	@echo "\n${HR}"
	@echo "Compiling and Compressing LESS with Recess ..."
	@echo "${hr}\n"
	recess --compile ${TB_LAYOUT_LESS} > ${TB_LAYOUT}
	recess --compress ${TB_LAYOUT_LESS} > ${TB_LAYOUT_MIN}
	recess --compile ${TB_SYTLES_LESS} > ${TB_SYTLES}
	recess --compress ${TB_SYTLES_LESS} > ${TB_SYTLES_MIN}
	recess --compile ${SIGNAL_LESS} > ${SIGNAL}
	recess --compress ${SIGNAL_LESS} > ${SIGNAL_MIN}
	recess --compile ${IE7_LESS} > ${IE7_SYTLES}
	recess --compress ${IE7_LESS} > ${IE7_MIN}	
	recess --compile ${PRINT_LESS} > ${PRINT_SYTLES}
	recess --compress ${PRINT_LESS} > ${PRINT_MIN}	
	recess --compile ${KENDO_TB_LESS} > ${KENDO_TB}
	recess --compress ${KENDO_TB_LESS} > ${KENDO_TB_MIN}
	@recess --compile ${BOOTSTRAP_LESS} > ${BOOTSTRAP}
	@recess --compress ${BOOTSTRAP_LESS} > ${BOOTSTRAP_MIN}
	@recess --compile ${BOOTSTRAP_RESPONSIVE_LESS} > ${BOOTSTRAP_RESPONSIVE}
	@recess --compress ${BOOTSTRAP_RESPONSIVE_LESS} > ${BOOTSTRAP_RESPONSIVE_MIN}
	@echo "\n${hr}"
	@echo "Compiling and Compressing LESS with Recess...               ${CHECK} Done"
	@echo "${HR}"

FIND_INSPECTJSFILES = ${shell find $(dir) -type f \( -name '*.js' ! -name '*.min.js' ! -name '*-min.js' \)}
INSPECTJSDIRS := .
INSPECTJSFILES := $(foreach dir, $(INSPECTJSDIRS), $(FIND_INSPECTJSFILES))

inspect:
	@echo "\n${HR}"
	@echo "Inspecting JS using jshint"
	@echo "${hr}"
	for file in $(INSPECTJSFILES); do \
		jshint --config .jshintrc $$file ;\
	done
	@jshint js/data*.js js/monitor*.js --config .jshintrc
	@jshint resources/scripts/*.js --config .jshintrc
	@jshint resources/scripts/plan/*.js --config .jshintrc
	@jshint resources/scripts/data/*.js --config .jshintrc
	@jshint resources/scripts/monitor/*.js --config .jshintrc
	@jshint resources/scripts/configure/configure/*.js --config .jshintrc
	find iondb/media/ -type f \( -iname "*.js" ! -iname "*.min.js" ! -iname "*-min.js" \) | xargs jshint --config iondb/media/.jshintrc
	@echo "\n${hr}"
	@echo "Running JSHint on javascript...             ${CHECK} Done"
	@echo "${HR}"

minify-js: | bulk-minify-js
	@echo "\n${HR}"
	@echo "Minify'ing Torrent Browser Bootstrap JS"
	@echo "${hr}"
	@uglifyjs -nc resources/bootstrap/js/bootstrap.js > resources/bootstrap/js/bootstrap.min.tmp.js
	@echo "/**\n* Bootstrap.js v2.2.1 by @fat & @mdo\n* Copyright 2012 Twitter, Inc.\n* http://www.apache.org/licenses/LICENSE-2.0.txt\n*/" > resources/bootstrap/js/copyright.js
	@cat resources/bootstrap/js/copyright.js resources/bootstrap/js/bootstrap.min.tmp.js > resources/bootstrap/js/bootstrap.min.js
	@rm resources/bootstrap/js/bootstrap.min.tmp.js resources/bootstrap/js/copyright.js
	@echo "\n${hr}"
	@echo "Minify'ing Torrent Browser Bootstrap JS.               ${CHECK} Done"
	@echo "${HR}"
	
	@echo "\n${HR}"
	@echo "Minify'ing Torrent Browser custom Kendo minified JS file" 
	@echo "${hr}"
	@echo "/*\n * Kendo UI Web v2012.1.322 (http://kendoui.com)\n* Copyright 2012 Telerik AD. All rights reserved.\n*\n* Kendo UI Web commercial licenses may be obtained at http://kendoui.com/web-license\n* If you do not own a commercial license, this file shall be governed by the\n* GNU General Public License (GPL) version 3.\n* For GPL requirements, please review: http://www.gnu.org/copyleft/gpl.html\n*/" > resources/scripts/kendo.custom.min.js 
	@cat resources/kendo/src/js/kendo.core.js \
		resources/kendo/src/js/kendo.data.js \
		resources/kendo/src/js/kendo.data.odata.js \
		resources/kendo/src/js/kendo.pager.js \
		resources/kendo/src/js/kendo.resizable.js \
		resources/kendo/src/js/kendo.sortable.js \
		resources/kendo/src/js/kendo.grid.js > resources/scripts/kendo.custom.js
	@uglifyjs -nc resources/scripts/kendo.custom.js >> resources/scripts/kendo.custom.min.js
	@echo "\n${hr}"
	@echo "Minify'ing Torrent Browser custom Kendo minified JS file...               ${CHECK} Done"
	@echo "${HR}"

	@echo "\n${HR}"
	@echo "Minify'ing random Torrent Browser JS"
	@uglifyjs -nc runliveness.js > runliveness.min.js 
	@uglifyjs -nc tbHelper.js > tbHelper.min.js 
	@echo "Minify'ing random Torrent Browser JS...               ${CHECK} Done"
	@echo "${HR}"

DIRS:= js resources/scripts resources/bootstrap-modal/js resources/uni-form/js
FIND_FILES = ${shell find $(dir) -maxdepth 1 -type f \( -name '*.js' ! -name '*.min.js' ! -name '*-min.js' \)}
JSFILES := $(foreach dir, $(DIRS), $(FIND_FILES))

bulk-minify-js:
	@echo "\n${HR}"
	@echo "Bulk Minify Process"
	@echo "${hr}"
	@echo "Processing files: $(JSFILES)"
	for file in $(JSFILES); do \
		ONAME=`echo "$$file" | rev | cut -d'.' -f1 --complement | rev`;\
		ONAME="$$ONAME.min.js";\
        echo "uglifying $$file to $$ONAME";\
        uglifyjs -o "$$ONAME" $$file ;\
	done
	@echo "\n${hr}"
	@echo "Bulk Minify Process...               ${CHECK} Done"
	@echo "${HR}"
	
.PHONY: all minify-js
