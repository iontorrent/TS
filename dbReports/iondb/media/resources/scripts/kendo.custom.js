/*
* Kendo UI Web v2012.1.322 (http://kendoui.com)
* Copyright 2012 Telerik AD. All rights reserved.
*
* Kendo UI Web commercial licenses may be obtained at http://kendoui.com/web-license
* If you do not own a commercial license, this file shall be governed by the
* GNU General Public License (GPL) version 3.
* For GPL requirements, please review: http://www.gnu.org/copyleft/gpl.html
*/
;(function($, undefined) {
    /**
     * @name kendo
     * @namespace This object contains all code introduced by the Kendo project, plus helper functions that are used across all widgets.
     */
    var kendo = window.kendo = window.kendo || {},
        extend = $.extend,
        each = $.each,
        proxy = $.proxy,
        isArray = $.isArray,
        noop = $.noop,
        isFunction = $.isFunction,
        math = Math,
        Template,
        JSON = window.JSON || {},
        support = {},
        boxShadowRegExp = /(\d+?)px\s*(\d+?)px\s*(\d+?)px\s*(\d+?)?/i,
        FUNCTION = "function",
        STRING = "string",
        NUMBER = "number",
        OBJECT = "object",
        NULL = "null",
        BOOLEAN = "boolean",
        UNDEFINED = "undefined",
        getterCache = {},
        setterCache = {},
        globalize = window.Globalize;

    function Class() {}

    Class.extend = function(proto) {
        var base = function() {},
            member,
            that = this,
            subclass = proto && proto.init ? proto.init : function () {
                that.apply(this, arguments);
            },
            fn;

        base.prototype = that.prototype;
        fn = subclass.fn = subclass.prototype = new base;

        for (member in proto) {
            if (typeof proto[member] === OBJECT && !(proto[member] instanceof Array) && proto[member] !== null) {
                // Merge object members
                fn[member] = extend(true, {}, base.prototype[member], proto[member]);
            } else {
                fn[member] = proto[member];
            }
        }

        fn.constructor = subclass;
        subclass.extend = that.extend;

        return subclass;
    };

    var Observable = Class.extend(/** @lends kendo.Observable.prototype */{
        /**
         * Creates an observable instance.
         * @constructs
         * @class Represents a class that can trigger events, along with methods that subscribe handlers to these events.
         */
        init: function() {
            this._events = {};
        },

        bind: function(eventName, handlers, one) {
            var that = this,
                idx,
                eventNames = typeof eventName === STRING ? [eventName] : eventName,
                length,
                original,
                handler,
                handlersIsFunction = typeof handlers === FUNCTION,
                events;

            for (idx = 0, length = eventNames.length; idx < length; idx++) {
                eventName = eventNames[idx];

                handler = handlersIsFunction ? handlers : handlers[eventName];

                if (handler) {
                    if (one) {
                        original = handler;
                        handler = function() {
                            that.unbind(eventName, handler);
                            original.apply(that, arguments);
                        }
                    }
                    events = that._events[eventName] = that._events[eventName] || [];
                    events.push(handler);
                }
            }

            return that;
        },

        one: function(eventNames, handlers) {
            return this.bind(eventNames, handlers, true);
        },

        trigger: function(eventName, e) {
            var that = this,
                events = that._events[eventName],
                idx,
                isDefaultPrevented = false;

            if (events) {
                e = e || {};

                e.sender = that;

                e.preventDefault = function () {
                    isDefaultPrevented = true;
                }

                e.isDefaultPrevented = function() {
                    return isDefaultPrevented;
                }

                //Do not cache the length of the events array as removing events attached through one will fail
                for (idx = 0; idx < events.length; idx++) {
                    events[idx].call(that, e);
                }
            }

            return isDefaultPrevented;
        },

        unbind: function(eventName, handler) {
            var that = this,
                events = that._events[eventName],
                idx,
                length;

            if (events) {
                if (handler) {
                    for (idx = 0, length = events.length; idx < length; idx++) {
                        if (events[idx] === handler) {
                            events.splice(idx, 1);
                        }
                    }
                } else {
                    that._events[eventName] = [];
                }
            }

            return that;
        }
    });

    /**
     * @name kendo.Template.Description
     *
     * @section
     * <p>
     *  Templates offer way of creating HTML chunks. Options such as HTML encoding and compilation for optimal
     *  performance are available.
     * </p>
     *
     * @exampleTitle Basic template
     * @example
     * var inlineTemplate = kendo.template("Hello, #= firstName # #= lastName #");
     * var inlineData = { firstName: "John", lastName: "Doe" };
     * $("#inline").html(inlineTemplate(inlineData));
     *
     * @exampleTitle Output:
     * @example
     * Hello, John Doe!
     *
     * @exampleTitle Encoding HTML
     * @example
     * var encodingTemplate = kendo.template("HTML tags are encoded as follows: ${ html }");
     * var encodingData = { html: "<strong>lorem ipsum</strong>" };
     * $("#encoding").html(encodingTemplate(encodingData));
     *
     * @exampleTitle Output:
     * @example
     * HTML tags are encoded as follows: <strong>lorem ipsum</strong>
     *
     */

     function compilePart(part, stringPart) {
         if (stringPart) {
             return "'" +
                 part.split("'").join("\\'")
                 .replace(/\n/g, "\\n")
                 .replace(/\r/g, "\\r")
                 .replace(/\t/g, "\\t")
                 + "'";
         } else {
             var first = part.charAt(0),
                 rest = part.substring(1);

             if (first === "=") {
                 return "+(" + rest + ")+";
             } else if (first === ":") {
                 return "+e(" + rest + ")+";
             } else {
                 return ";" + part + ";o+=";
             }
         }
     }

    var argumentNameRegExp = /^\w+/,
        encodeRegExp = /\${([^}]*)}/g,
        escapedCurlyRegExp = /\\}/g,
        curlyRegExp = /__CURLY__/g,
        escapedSharpRegExp = /\\#/g,
        sharpRegExp = /__SHARP__/g;

    /**
     * @name kendo.Template
     * @namespace
     */
    Template = /** @lends kendo.Template */ {
        paramName: "data", // name of the parameter of the generated template
        useWithBlock: true, // whether to wrap the template in a with() block
        /**
         * Renders a template for each item of the data.
         * @ignore
         * @name kendo.Template.render
         * @static
         * @function
         * @param {String} [template] The template that will be rendered
         * @param {Array} [data] Data items
         * @returns {String} The rendered template
         */
        render: function(template, data) {
            var idx,
                length,
                html = "";

            for (idx = 0, length = data.length; idx < length; idx++) {
                html += template(data[idx]);
            }

            return html;
        },
        /**
         * Compiles a template to a function that builds HTML. Useful when a template will be used several times.
         * @ignore
         * @name kendo.Template.compile
         * @static
         * @function
         * @param {String} [template] The template that will be compiled
         * @param {Object} [options] Compilation options
         * @returns {Function} The compiled template
         */
        compile: function(template, options) {
            var settings = extend({}, this, options),
                paramName = settings.paramName,
                argumentName = paramName.match(argumentNameRegExp)[0],
                useWithBlock = settings.useWithBlock,
                functionBody = "var o,e=kendo.htmlEncode;",
                parts,
                part,
                idx;

            if (isFunction(template)) {
                if (template.length === 2) {
                    //looks like jQuery.template
                    return function(d) {
                        return template($, { data: d }).join("");
                    }
                }
                return template;
            }

            functionBody += useWithBlock ? "with(" + paramName + "){" : "";

            functionBody += "o=";

            parts = template
                .replace(escapedCurlyRegExp, "__CURLY__")
                .replace(encodeRegExp, "#=e($1)#")
                .replace(curlyRegExp, "}")
                .replace(escapedSharpRegExp, "__SHARP__")
                .split("#");

            for (idx = 0; idx < parts.length; idx ++) {
                functionBody += compilePart(parts[idx], idx % 2 === 0);
            }

            functionBody += useWithBlock ? ";}" : ";";

            functionBody += "return o;";

            functionBody = functionBody.replace(sharpRegExp, "#");

            try {
                return new Function(argumentName, functionBody);
            } catch(e) {
                throw new Error(kendo.format("Invalid template:'{0}' Generated code:'{1}'", template, functionBody));
            }
        }
    };

    //JSON stringify
(function() {
    var cx = /[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,
        escapable = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,
        gap,
        indent,
        meta = {
            "\b": "\\b",
            "\t": "\\t",
            "\n": "\\n",
            "\f": "\\f",
            "\r": "\\r",
            "\"" : '\\"',
            "\\": "\\\\"
        },
        rep,
        formatters,
        toString = {}.toString,
        hasOwnProperty = {}.hasOwnProperty;

    if (typeof Date.prototype.toJSON !== FUNCTION) {

        /** @ignore */
        Date.prototype.toJSON = function (key) {
            var that = this;

            return isFinite(that.valueOf()) ?
                that.getUTCFullYear()     + "-" +
                pad(that.getUTCMonth() + 1) + "-" +
                pad(that.getUTCDate())      + "T" +
                pad(that.getUTCHours())     + ":" +
                pad(that.getUTCMinutes())   + ":" +
                pad(that.getUTCSeconds())   + "Z" : null;
        };

        String.prototype.toJSON = Number.prototype.toJSON = /** @ignore */ Boolean.prototype.toJSON = function (key) {
            return this.valueOf();
        };
    }

    function quote(string) {
        escapable.lastIndex = 0;
        return escapable.test(string) ? "\"" + string.replace(escapable, function (a) {
            var c = meta[a];
            return typeof c === STRING ? c :
                "\\u" + ("0000" + a.charCodeAt(0).toString(16)).slice(-4);
        }) + "\"" : "\"" + string + "\"";
    }

    function str(key, holder) {
        var i,
            k,
            v,
            length,
            mind = gap,
            partial,
            value = holder[key],
            type;

        if (value && typeof value === OBJECT && typeof value.toJSON === FUNCTION) {
            value = value.toJSON(key);
        }

        if (typeof rep === FUNCTION) {
            value = rep.call(holder, key, value);
        }

        type = typeof value;
        if (type === STRING) {
            return quote(value);
        } else if (type === NUMBER) {
            return isFinite(value) ? String(value) : NULL;
        } else if (type === BOOLEAN || type === NULL) {
            return String(value);
        } else if (type === OBJECT) {
            if (!value) {
                return NULL;
            }
            gap += indent;
            partial = [];
            if (toString.apply(value) === "[object Array]") {
                length = value.length;
                for (i = 0; i < length; i++) {
                    partial[i] = str(i, value) || NULL;
                }
                v = partial.length === 0 ? "[]" : gap ?
                    "[\n" + gap + partial.join(",\n" + gap) + "\n" + mind + "]" :
                    "[" + partial.join(",") + "]";
                gap = mind;
                return v;
            }
            if (rep && typeof rep === OBJECT) {
                length = rep.length;
                for (i = 0; i < length; i++) {
                    if (typeof rep[i] === STRING) {
                        k = rep[i];
                        v = str(k, value);
                        if (v) {
                            partial.push(quote(k) + (gap ? ": " : ":") + v);
                        }
                    }
                }
            } else {
                for (k in value) {
                    if (hasOwnProperty.call(value, k)) {
                        v = str(k, value);
                        if (v) {
                            partial.push(quote(k) + (gap ? ": " : ":") + v);
                        }
                    }
                }
            }

            v = partial.length === 0 ? "{}" : gap ?
                "{\n" + gap + partial.join(",\n" + gap) + "\n" + mind + "}" :
                "{" + partial.join(",") + "}";
            gap = mind;
            return v;
        }
    }

    if (typeof JSON.stringify !== FUNCTION) {
        JSON.stringify = function (value, replacer, space) {
            var i;
            gap = "";
            indent = "";

            if (typeof space === NUMBER) {
                for (i = 0; i < space; i += 1) {
                    indent += " ";
                }

            } else if (typeof space === STRING) {
                indent = space;
            }

            rep = replacer;
            if (replacer && typeof replacer !== FUNCTION && (typeof replacer !== OBJECT || typeof replacer.length !== NUMBER)) {
                throw new Error("JSON.stringify");
            }

            return str("", {"": value});
        };
    }
})();

// Date and Number formatting
(function() {
    var formatRegExp = /{(\d+)(:[^\}]+)?}/g,
        dateFormatRegExp = /dddd|ddd|dd|d|MMMM|MMM|MM|M|yyyy|yy|HH|H|hh|h|mm|m|fff|ff|f|tt|ss|s|"[^"]*"|'[^']*'/g,
        standardFormatRegExp =  /^(n|c|p|e)(\d*)$/i,
        literalRegExp = /["'].*?["']/g,
        EMPTY = "",
        POINT = ".",
        COMMA = ",",
        SHARP = "#",
        ZERO = "0",
        PLACEHOLDER = "??",
        EN = "en-US";

    //cultures
    kendo.cultures = {"en-US" : {
        name: EN,
        numberFormat: {
            pattern: ["-n"],
            decimals: 2,
            ",": ",",
            ".": ".",
            groupSize: [3],
            percent: {
                pattern: ["-n %", "n %"],
                decimals: 2,
                ",": ",",
                ".": ".",
                groupSize: [3],
                symbol: "%"
            },
            currency: {
                pattern: ["($n)", "$n"],
                decimals: 2,
                ",": ",",
                ".": ".",
                groupSize: [3],
                symbol: "$"
            }
        },
        calendars: {
            standard: {
                days: {
                    names: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
                    namesAbbr: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
                    namesShort: [ "Su", "Mo", "Tu", "We", "Th", "Fr", "Sa" ]
                },
                months: {
                    names: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
                    namesAbbr: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                },
                AM: [ "AM", "am", "AM" ],
                PM: [ "PM", "pm", "PM" ],
                patterns: {
                    d: "M/d/yyyy",
                    D: "dddd, MMMM dd, yyyy",
                    F: "dddd, MMMM dd, yyyy h:mm:ss tt",
                    g: "M/d/yyyy h:mm tt",
                    G: "M/d/yyyy h:mm:ss tt",
                    m: "MMMM dd",
                    M: "MMMM dd",
                    s: "yyyy'-'MM'-'ddTHH':'mm':'ss",
                    t: "h:mm tt",
                    T: "h:mm:ss tt",
                    u: "yyyy'-'MM'-'dd HH':'mm':'ss'Z'",
                    y: "MMMM, yyyy",
                    Y: "MMMM, yyyy"
                },
                "/": "/",
                ":": ":",
                firstDay: 0
            }
        }
    }};

    /**
     * @name kendo.ui.Globalization
     * @namespace
     */
     /**
     * @name kendo.ui.Globalization.Description
     *
     * @section Globalization is the process of designing and developing an
     * application that works in multiple cultures. The culture defines specific information
     * for the number formats, week and month names, date and time formats and etc.
     *
     * @section Kendo exposes <strong><em>culture(cultureName)</em></strong> method which allows to select the culture
     * script coresponding to the "culture name". kendo.culture() method uses the passed culture name
     * to select culture from the culture scripts that you have included and then sets the current culture.
     * If there is no such culture, the default one is used.
     *
     * <h3>Define current culture settings</h3>
     *
     * @exampleTitle Include culture scripts and select culture
     * @example
     *
     * <script src="jquery.js" ></script>
     * <script src="kendo.all.min.js"></script>
     * <script src="kendo.culture.en-GB.js"></script>
     * <script type="text/javascript">
     *    //set current culture to the "en-GB" culture script.
     *    kendo.culture("en-GB");
     * </script>
     *
     * @exampleTitle Get current culture
     * @example
     * var cultureInfo = kendo.culture();
     *
     * @section
     * <h3>Format number or date object</h3>
     *
     * Kendo exposes methods which can format number or date object using specific format string and the current specified culture:
     * @section
     * <h4><code>kendo.toString(object, format)</code> - returns a string representation of the current object using specific format.</h4>
     * @exampleTitle Formats number and date objects
     * @example
     * //format number using standard number format
     * kendo.toString(10.12, "n"); //10.12
     * kendo.toString(10.12, "n0"); //10
     * kendo.toString(10.12, "n5"); //10.12000
     * kendo.toString(10.12, "c"); //$10.12
     * kendo.toString(0.12, "p"); //12.00 %
     * //format number using custom number format
     * kendo.toString(19.12, "00##"); //0019
     * //format date
     * kendo.toString(new Date(2010, 9, 5), "yyyy/MM/dd" ); // "2010/10/05"
     * kendo.toString(new Date(2010, 9, 5), "dddd MMMM d, yyyy" ); // "Tuesday October 5, 2010"
     * kendo.toString(new Date(2010, 10, 10, 22, 12), "hh:mm tt" ); // "10:12 PM"
     *
     * @section
     * <h4><code>kendo.format</code> - replaces each format item in a specified string with the text equivalent of a corresponding object's value.</h4>
     *  @exampleTitle String format
     *  @example
     *  kendo.format("{0} - {1}", 12, 24); //12 - 24
     *  kendo.format("{0:c} - {1:c}", 12, 24); //$12.00 - $24.00
     *
     * @section
     * <h3>Parsing a string</h3>
     *
     * Kendo exposes methods which converts the specified string to date or number object:
     * <ol>
     *    <li>
     *       <code>kendo.parseInt(string, [culture])</code> - converts a string to a whole number using the specified culture (current culture by default).
     *        @exampleTitle Parse string to integer
     *        @example
     *
     *        //assumes that current culture defines decimal separator as "."
     *        kendo.parseInt("12.22"); //12
     *
     *        //assumes that current culture defines decimal separator as ",", group separator as "." and currency symbol as "€"
     *        kendo.parseInt("1.212,22 €"); //1212
     *    </li>
     *    <li>
     *       <code>kendo.parseFloat(string, [culture])</code> - converts a string to a number with floating point using the specified culture (current culture by default).
     *        @exampleTitle Parse string to float
     *        @example
     *
     *        //assumes that current culture defines decimal separator as "."
     *        kendo.parseFloat("12.22"); //12.22
     *
     *        //assumes that current culture defines decimal separator as ",", group separator as "." and currency symbol as "€"
     *        kendo.parseFloat("1.212,22 €"); //1212.22
     *    </li>
     *    <li>
     *       <code>kendo.parseDate(string, [formats], [culture])</code> - converts a string to a JavaScript Date object, taking into account the given format/formats (or the given culture's set of default formats).
     *       Current culture is used if one is not specified.
     *        @exampleTitle Parse string to float
     *        @example
     *
     *        //current culture is "en-US"
     *        kendo.parseDate("12/22/2000"); //Fri Dec 22 2000
     *        kendo.parseDate("2000/12/22", "yyyy/MM/dd"); //Fri Dec 22 2000
     *    </li>
     * </ol>
     *
     * @section
     * <h3>Number formatting</h3>
     * The purpose of number formatting is to convert Number object to a human readable string using culture's specific settings. <code>kendo.format</code> and <code>kendo.toString</code>
     * methods support standard and custom numeric formats:
     * <h4>Standard numeric formats</h4>
     *<strong>n</strong> for number
     *       @exampleTitle Formatting using "n" format
     *       @example
     *       kendo.culture("en-US");
     *       kendo.toString(1234.567, "n"); //1,234.57
     *
     *       kendo.culture("de-DE");
     *       kendo.toString(1234.567, "n3"); //1.234,567
     *@section
     *<strong>c</strong> for currency
     *       @exampleTitle Formatting using "c" format
     *       @example
     *       kendo.culture("en-US");
     *       kendo.toString(1234.567, "c"); //$1,234.57
     *
     *       kendo.culture("de-DE");
     *       kendo.toString(1234.567, "c3"); //1.234,567 €
     *@section
     *<strong>p</strong> for percentage (number is multiplied by 100)
     *       @exampleTitle Formatting using "p" format
     *       @example
     *       kendo.culture("en-US");
     *       kendo.toString(0.222, "p"); //22.20 %
     *
     *       kendo.culture("de-DE");
     *       kendo.toString(0.22, "p3"); //22.000 %
     *@section
     *<strong>e</strong> for exponential
     *       @exampleTitle Formatting using "e" format
     *       @example
     *       kendo.toString(0.122, "e"); //1.22e-1
     *       kendo.toString(0.122, "e4"); //1.2200e-1
     *
     * @section
     * <h4>Custom numeric formats</h4>
     * You can create custom numeric format string using one or more custom numeric specifiers. Custom numeric format string is any tha is not a standard numeric format.
     * <div class="details-list">
     *   <h4 class="details-title">Format specifiers</h4>
     *   <dl>
     *     <dt>
     *       "0" - zero placeholder
     *     </dt>
     *     <dd>Replaces the zero with the corresponding digit if one is present; otherwise, zero appears in the result string - <code>kendo.toString(1234.5678, "00000") -> 01235</code></dd>
     *     <dt>
     *       "#" - digit placeholder
     *     </dt>
     *     <dd>Replaces the pound sign with the corresponding digit if one is present; otherwise, no digit appears in the result string - <code>kendo.toString(1234.5678, "#####") -> 1235</code></dd>
     *     <dt>
     *       "." - Decimal placeholder
     *     </dt>
     *     <dd>Determines the location of the decimal separator in the result string - <code>kendo.tostring(0.45678, "0.00") -> 0.46 </code>(en-us)</dd>
     *     <dt>
     *       "," - group separator placeholder
     *     </dt>
     *     <dd>Insert localized group separator between each group - <code>kendo.tostring(12345678, "##,#") -> 12,345,678</code>(en-us)</dd>
     *     <dt>
     *       "%" - percentage placeholder
     *     </dt>
     *     <dd>Multiplies a number by 100 and inserts a localized percentage symbol in the result string</dd>
     *     <dt>
     *       "e" - exponential notation
     *     </dt>
     *     <dd><code>kendo.toString(0.45678, "e0") -> 5e-1</code></dd>
     *     <dt>
     *       ";" - section separator
     *     </dt>
     *     <dd>Defines sections wih separate format strings for positive, negative, and zero numbers</dd>
     *     <dt>
     *       "string"/'string' - Literal string delimiter
     *     </dt>
     *     <dd>Indicates that the enclosed characters should be copied to the result string</dd>
     *   </dl>
     * </div>
     *
     * @section
     * <h3>Date formatting</h3>
     * The purpose of date formatting is to convert Date object to a human readable string using culture's specific settings. <code>kendo.format</code> and <code>kendo.toString</code>
     * methods support standard and custom date formats:
     * <h4>Standard date formats</h4>
     * <div class="details-list">
     *   <h4 class="details-title">Format specifiers</h4>
     *   <dl>
     *     <dt>
     *       "d" - short date pattern
     *     </dt>
     *     <dd><code>kendo.toString(new Date(2000, 10, 6), "d") -> 11/6/2000</code></dd>
     *     <dt>
     *       "D" - long date pattern
     *     </dt>
     *     <dd><code>kendo.toString(new Date(2000, 10, 6), "D") -> Monday, November 06, 2000</code></dd>
     *     <dt>
     *       "F" - Full date/time pattern
     *     </dt>
     *     <dd><code>kendo.toString(new Date(2000, 10, 6), "D") -> Monday, November 06, 2000 12:00:00 AM</code></dd>
     *     <dt>
     *       "g" - General date/time pattern (short time)
     *     </dt>
     *     <dd><code>kendo.toString(new Date(2000, 10, 6), "g") -> 11/6/2000 12:00 AM</code></dd>
     *     <dt>
     *       "G" - General date/time pattern (long time)
     *     </dt>
     *     <dd><code>kendo.toString(new Date(2000, 10, 6), "G") -> 11/6/2000 12:00:00 AM</code></dd>
     *     <dt>
     *       "M/m" - Month/day pattern
     *     </dt>
     *     <dd><code>kendo.toString(new Date(2000, 10, 6), "m") -> November 06</code></dd>
     *     <dt>
     *       "u" - Universal sortable date/time pattern
     *     </dt>
     *     <dd><code>kendo.toString(new Date(2000, 10, 6), "u") -> 2000-11-06 00:00:00Z</code></dd>
     *     <dt>
     *       "Y/y" - Year/month pattern
     *     </dt>
     *     <dd><code>kendo.toString(new Date(2000, 10, 6), "y") -> November, 2000</code></dd>
     *   </dl>
     * </div>
     *
     *@section
     * <h4>Custom date formats</h4>
     * <div class="details-list">
     *   <h4 class="details-title">Format specifiers</h4>
     *   <dl>
     *     <dt>
     *       "d"
     *     </dt>
     *     <dd>The day of the month, from 1 through 31</dd>
     *     <dt>
     *       "dd"
     *     </dt>
     *     <dd>The day of the month, from 01 through 31.</dd>
     *     <dt>
     *       "ddd"
     *     </dt>
     *     <dd>iThe abbreviated name of the day of the week</dd>
     *     <dt>
     *       "dddd"
     *     </dt>
     *     <dd>The full name of the day of the week</dd>
     *     <dt>
     *       "f"
     *     </dt>
     *     <dd>The tenths of a second in a date and time value</dd>
     *     <dt>
     *       "ff"
     *     </dt>
     *     <dd>The hundredths of a second in a date and time value</dd>
     *     <dt>
     *       "fff"
     *     </dt>
     *     <dd>The milliseconds in a date and time value</dd>
     *     <dt>
     *       "M"
     *     </dt>
     *     <dd>The month, from 1 through 12</dd>
     *     <dt>
     *       "MM"
     *     </dt>
     *     <dd>The month, from 01 through 12</dd>
     *     <dt>
     *       "MMM"
     *     </dt>
     *     <dd>The abbreviated name of the month</dd>
     *     <dt>
     *       "MMMM"
     *     </dt>
     *     <dd>The full name of the month</dd>
     *     <dt>
     *       "h"
     *     </dt>
     *     <dd>The hour, using a 12-hour clock from 1 to 12</dd>
     *     <dt>
     *       "hh"
     *     </dt>
     *     <dd>The hour, using a 12-hour clock from 01 to 12</dd>
     *     <dt>
     *       "H"
     *     </dt>
     *     <dd>The hour, using a 24-hour clock from 1 to 23</dd>
     *     <dt>
     *       "HH"
     *     </dt>
     *     <dd>The hour, using a 24-hour clock from 01 to 23</dd>
     *     <dt>
     *       "m"
     *     </dt>
     *     <dd>The minute, from 0 through 59</dd>
     *     <dt>
     *       "mm"
     *     </dt>
     *     <dd>The minute, from 00 through 59</dd>
     *     <dt>
     *       "s"
     *     </dt>
     *     <dd>The second, from 0 through 59</dd>
     *     <dt>
     *       "ss"
     *     </dt>
     *     <dd>The second, from 00 through 59</dd>
     *     <dt>
     *       "tt"
     *     </dt>
     *     <dd>The AM/PM designator</dd>
     *   </dl>
     * </div>
     *
     * @section
     * <p><h3>Widgets that depend on current culture are:</h3>
     *    <ul>
     *        <li> Calendar </li>
     *        <li> DatePicker </li>
     *        <li> TimePicker </li>
     *        <li> NumericTextBox </li>
     *    </ul>
     * </p>
     */

    kendo.culture = function(cultureName) {
        if (cultureName !== undefined) {
            var cultures = kendo.cultures,
                culture = cultures[cultureName] || cultures[EN];

            culture.calendar = culture.calendars.standard;
            cultures.current = culture;
        } else {
            return kendo.cultures.current;
        }
    };

    //set current culture to en-US.
    kendo.culture(EN);

    function pad(number) {
        return number < 10 ? "0" + number : number;
    }

    function formatDate(date, format) {
        var calendar = kendo.cultures.current.calendar,
            days = calendar.days,
            months = calendar.months;

        format = calendar.patterns[format] || format;

        return format.replace(dateFormatRegExp, function (match) {
            var result;

            if (match === "d") {
                result = date.getDate();
            } else if (match === "dd") {
                result = pad(date.getDate());
            } else if (match === "ddd") {
                result = days.namesAbbr[date.getDay()];
            } else if (match === "dddd") {
                result = days.names[date.getDay()];
            } else if (match === "M") {
                result = date.getMonth() + 1;
            } else if (match === "MM") {
                result = pad(date.getMonth() + 1);
            } else if (match === "MMM") {
                result = months.namesAbbr[date.getMonth()];
            } else if (match === "MMMM") {
                result = months.names[date.getMonth()];
            } else if (match === "yy") {
                result = pad(date.getFullYear() % 100);
            } else if (match === "yyyy") {
                result = date.getFullYear();
            } else if (match === "h" ) {
                result = date.getHours() % 12 || 12
            } else if (match === "hh") {
                result = pad(date.getHours() % 12 || 12);
            } else if (match === "H") {
                result = date.getHours();
            } else if (match === "HH") {
                result = pad(date.getHours());
            } else if (match === "m") {
                result = date.getMinutes();
            } else if (match === "mm") {
                result = pad(date.getMinutes());
            } else if (match === "s") {
                result = date.getSeconds();
            } else if (match === "ss") {
                result = pad(date.getSeconds());
            } else if (match === "f") {
                result = math.floor(date.getMilliseconds() / 100);
            } else if (match === "ff") {
                result = math.floor(date.getMilliseconds() / 10);
            } else if (match === "fff") {
                result = date.getMilliseconds();
            } else if (match === "tt") {
                result = date.getHours() < 12 ? calendar.AM[0] : calendar.PM[0]
            }

            return result !== undefined ? result : match.slice(1, match.length - 1);
        });
    }

    //number formatting
    function formatNumber(number, format) {
        var culture = kendo.cultures.current,
            numberFormat = culture.numberFormat,
            groupSize = numberFormat.groupSize[0],
            groupSeparator = numberFormat[COMMA],
            decimal = numberFormat[POINT],
            precision = numberFormat.decimals,
            pattern = numberFormat.pattern[0],
            literals = [],
            symbol,
            isCurrency, isPercent,
            customPrecision,
            formatAndPrecision,
            negative = number < 0,
            integer,
            fraction,
            integerLength,
            fractionLength,
            replacement = EMPTY,
            value = EMPTY,
            idx,
            length,
            ch,
            decimalIndex,
            sharpIndex,
            zeroIndex,
            start = -1,
            end;

        //return empty string if no number
        if (number === undefined) {
            return EMPTY;
        }

        if (!isFinite(number)) {
            return number;
        }

        //if no format then return number.toString() or number.toLocaleString() if culture.name is not defined
        if (!format) {
            return culture.name.length ? number.toLocaleString() : number.toString();
        }

        formatAndPrecision = standardFormatRegExp.exec(format);

        /* standard formatting */
        if (formatAndPrecision) {
            format = formatAndPrecision[1].toLowerCase();

            isCurrency = format === "c";
            isPercent = format === "p";

            if (isCurrency || isPercent) {
                //get specific number format information if format is currency or percent
                numberFormat = isCurrency ? numberFormat.currency : numberFormat.percent;
                groupSize = numberFormat.groupSize[0];
                groupSeparator = numberFormat[COMMA];
                decimal = numberFormat[POINT];
                precision = numberFormat.decimals;
                symbol = numberFormat.symbol;
                pattern = numberFormat.pattern[negative ? 0 : 1];
            }

            customPrecision = formatAndPrecision[2];

            if (customPrecision) {
                precision = +customPrecision;
            }

            //return number in exponential format
            if (format === "e") {
                return customPrecision ? number.toExponential(precision) : number.toExponential(); // toExponential() and toExponential(undefined) differ in FF #653438.
            }

            // multiply if format is percent
            if (isPercent) {
                number *= 100;
            }

            number = number.toFixed(precision);
            number = number.split(POINT);

            integer = number[0];
            fraction = number[1];

            //exclude "-" if number is negative.
            if (negative) {
                integer = integer.substring(1);
            }

            value = integer;
            integerLength = integer.length;

            //add group separator to the number if it is longer enough
            if (integerLength >= groupSize) {
                value = EMPTY;
                for (idx = 0; idx < integerLength; idx++) {
                    if (idx > 0 && (integerLength - idx) % groupSize === 0) {
                        value += groupSeparator;
                    }
                    value += integer.charAt(idx);
                }
            }

            if (fraction) {
                value += decimal + fraction;
            }

            if (format === "n" && !negative) {
                return value;
            }

            number = EMPTY;

            for (idx = 0, length = pattern.length; idx < length; idx++) {
                ch = pattern.charAt(idx);

                if (ch === "n") {
                    number += value;
                } else if (ch === "$" || ch === "%") {
                    number += symbol;
                } else {
                    number += ch;
                }
            }

            return number;
        }

        //custom formatting
        //
        //separate format by sections.
        format = format.split(";");
        if (negative && format[1]) {
            //make number positive and get negative format
            number = -number;
            format = format[1];
        } else if (number === 0) {
            //format for zeros
            format = format[2] || format[0];
            if (format.indexOf(SHARP) == -1 && format.indexOf(ZERO) == -1) {
                //return format if it is string constant.
                return format;
            }
        } else {
            format = format[0];
        }

        if (format.indexOf("'") > -1 || format.indexOf("\"") > -1) {
            format = format.replace(literalRegExp, function(match) {
                literals.push(match);
                return PLACEHOLDER;
            });
        }

        isCurrency = format.indexOf("$") != -1;
        isPercent = format.indexOf("%") != -1;

        //multiply number if the format has percent
        if (isPercent) {
            number *= 100;
        }

        if (isCurrency || isPercent) {
            //get specific number format information if format is currency or percent
            numberFormat = isCurrency ? numberFormat.currency : numberFormat.percent;
            groupSize = numberFormat.groupSize[0];
            groupSeparator = numberFormat[COMMA];
            decimal = numberFormat[POINT];
            precision = numberFormat.decimals;
            symbol = numberFormat.symbol;
        }

        decimalIndex = format.indexOf(POINT);
        length = format.length;

        if (decimalIndex != -1) {
            sharpIndex = format.lastIndexOf(SHARP);
            zeroIndex = format.lastIndexOf(ZERO);

            if (zeroIndex != -1) {
                value = number.toFixed(zeroIndex - decimalIndex);
                number = number.toString();
                number = number.length > value.length && sharpIndex > zeroIndex ? number : value;
            }
        } else {
            number = number.toFixed(0);
        }

        sharpIndex = format.indexOf(SHARP);
        zeroIndex = format.indexOf(ZERO);

        //define the index of the first digit placeholder
        if (sharpIndex == -1 && zeroIndex != -1) {
            start = zeroIndex;
        } else if (sharpIndex != -1 && zeroIndex == -1) {
            start = sharpIndex;
        } else {
            start = sharpIndex > zeroIndex ? zeroIndex : sharpIndex;
        }

        sharpIndex = format.lastIndexOf(SHARP);
        zeroIndex = format.lastIndexOf(ZERO);

        //define the index of the last digit placeholder
        if (sharpIndex == -1 && zeroIndex != -1) {
            end = zeroIndex;
        } else if (sharpIndex != -1 && zeroIndex == -1) {
            end = sharpIndex;
        } else {
            end = sharpIndex > zeroIndex ? sharpIndex : zeroIndex;
        }

        if (start == length) {
            end = start;
        }

        if (start != -1) {
            value = number.toString().split(POINT);
            integer = value[0];
            fraction = value[1] || EMPTY;

            integerLength = integer.length;
            fractionLength = fraction.length;

            //add group separator to the number if it is longer enough
            if (integerLength >= groupSize && format.indexOf(COMMA) != -1) {
                value = EMPTY;
                for (idx = 0; idx < integerLength; idx++) {
                    if (idx > 0 && (integerLength - idx) % groupSize === 0) {
                        value += groupSeparator;
                    }
                    value += integer.charAt(idx);
                }
                integer = value;
            }

            number = format.substring(0, start);

            for (idx = start; idx < length; idx++) {
                ch = format.charAt(idx);

                if (decimalIndex == -1) {
                    if (end - idx < integerLength) {
                        number += integer;
                        break;
                    }
                } else {
                    if (zeroIndex != -1 && zeroIndex < idx) {
                        replacement = EMPTY;
                    }

                    if ((decimalIndex - idx) <= integerLength && decimalIndex - idx > -1) {
                        number += integer;
                        idx = decimalIndex;
                    }

                    if (decimalIndex === idx) {
                        number += (fraction ? decimal : EMPTY) + fraction;
                        idx += end - decimalIndex + 1;
                        continue;
                    }
                }

                if (ch === ZERO) {
                    number += ch;
                    replacement = ch;
                } else if (ch === SHARP) {
                    number += replacement;
                } else if (ch === COMMA) {
                    continue;
                }
            }

            if (end >= start) {
                number += format.substring(end + 1);
            }

            //replace symbol placeholders
            if (isCurrency || isPercent) {
                value = EMPTY;
                for (idx = 0, length = number.length; idx < length; idx++) {
                    ch = number.charAt(idx);
                    value += (ch === "$" || ch === "%") ? symbol : ch;
                }
                number = value;
            }

            if (literals[0]) {
                length = literals.length;
                for (idx = 0; idx < length; idx++) {
                    number = number.replace(PLACEHOLDER, literals[idx]);
                }
            }
        }

        return number;
    }

    function toString(value, fmt) {
        if (fmt) {
            if (value instanceof Date) {
                return formatDate(value, fmt);
            } else if (typeof value === NUMBER) {
                return formatNumber(value, fmt);
            }
        }

        return value !== undefined ? value : "";
    }

    if (globalize) {
        toString = proxy(globalize.format, globalize);
    }

    kendo.format = function(fmt) {
        var values = arguments;

        return fmt.replace(formatRegExp, function(match, index, placeholderFormat) {
            var value = values[parseInt(index) + 1];

            return toString(value, placeholderFormat ? placeholderFormat.substring(1) : "");
        });
    };

    kendo.toString = toString;
    })();


(function() {

    var nonBreakingSpaceRegExp = /\u00A0/g,
        exponentRegExp = /[eE][-+]?[0-9]+/,
        formatsSequence = ["G", "g", "d", "F", "D", "y", "m", "T", "t"];

    function outOfRange(value, start, end) {
        return !(value >= start && value <= end);
    }

    function parseExact(value, format, culture) {
        if (!value) {
            return null;
        }

        var lookAhead = function (match) {
                var i = 0;
                while (format[idx] === match) {
                    i++;
                    idx++;
                }
                if (i > 0) {
                    idx -= 1;
                }
                return i;
            },
            getNumber = function(size) {
                var rg = new RegExp('^\\d{1,' + size + '}'),
                    match = value.substr(valueIdx, size).match(rg);

                if (match) {
                    match = match[0];
                    valueIdx += match.length;
                    return parseInt(match, 10);
                }
                return null;
            },
            getIndexByName = function (names) {
                var i = 0,
                    length = names.length,
                    name, nameLength;

                for (; i < length; i++) {
                    name = names[i];
                    nameLength = name.length;

                    if (value.substr(valueIdx, nameLength) == name) {
                        valueIdx += nameLength;
                        return i + 1;
                    }
                }
                return null;
            },
            checkLiteral = function() {
                if (value.charAt(valueIdx) == format[idx]) {
                    valueIdx++;
                }
            },
            calendar = culture.calendar,
            year = null,
            month = null,
            day = null,
            hours = null,
            minutes = null,
            seconds = null,
            milliseconds = null,
            idx = 0,
            valueIdx = 0,
            literal = false,
            date = new Date(),
            defaultYear = date.getFullYear(),
            shortYearCutOff = 30,
            ch, count, AM, PM, pmHour, length, pattern;

        if (!format) {
            format = "d"; //shord date format
        }

        //if format is part of the patterns get real format
        pattern = calendar.patterns[format];
        if (pattern) {
            format = pattern;
        }

        format = format.split("");
        length = format.length;

        for (; idx < length; idx++) {
            ch = format[idx];

            if (literal) {
                if (ch === "'") {
                    literal = false;
                } else {
                    checkLiteral();
                }
            } else {
                if (ch === "d") {
                    count = lookAhead("d");
                    day = count < 3 ? getNumber(2) : getIndexByName(calendar.days[count == 3 ? "namesAbbr" : "names"]);

                    if (day === null || outOfRange(day, 1, 31)) {
                        return null;
                    }
                } else if (ch === "M") {
                    count = lookAhead("M");
                    month = count < 3 ? getNumber(2) : getIndexByName(calendar.months[count == 3 ? 'namesAbbr' : 'names']);

                    if (month === null || outOfRange(month, 1, 12)) {
                        return null;
                    }
                    month -= 1; //because month is zero based
                } else if (ch === "y") {
                    count = lookAhead("y");
                    year = getNumber(count < 3 ? 2 : 4);
                    if (year === null) {
                        year = defaultYear;
                    }
                    if (year < shortYearCutOff) {
                        year = (defaultYear - defaultYear % 100) + year;
                    }
                } else if (ch === "h" ) {
                    lookAhead("h");
                    hours = getNumber(2);
                    if (hours == 12) {
                        hours = 0;
                    }
                    if (hours === null || outOfRange(hours, 0, 11)) {
                        return null;
                    }
                } else if (ch === "H") {
                    lookAhead("H");
                    hours = getNumber(2);
                    if (hours === null || outOfRange(hours, 0, 23)) {
                        return null;
                    }
                } else if (ch === "m") {
                    lookAhead("m");
                    minutes = getNumber(2);
                    if (minutes === null || outOfRange(minutes, 0, 59)) {
                        return null;
                    }
                } else if (ch === "s") {
                    lookAhead("s");
                    seconds = getNumber(2);
                    if (seconds === null || outOfRange(seconds, 0, 59)) {
                        return null;
                    }
                } else if (ch === "f") {
                    count = lookAhead("f");
                    milliseconds = getNumber(count);
                    if (milliseconds === null || outOfRange(milliseconds, 0, 999)) {
                        return null;
                    }
                } else if (ch === "t") {
                    count = lookAhead("t");
                    pmHour = getIndexByName(calendar.PM);
                } else if (ch === "'") {
                    checkLiteral();
                    literal = true;
                } else {
                    checkLiteral();
                }
            }
        }

        if (pmHour && hours < 12) {
            hours += 12;
        }

        if (day === null) {
            day = 1;
        }

        return new Date(year, month, day, hours, minutes, seconds, milliseconds);
    }

    kendo.parseDate = function(value, formats, culture) {
        if (value instanceof Date) {
            return value;
        }

        var idx = 0,
            date = null,
            length, property, patterns;

        if (!culture) {
            culture = kendo.culture();
        } else if (typeof culture === STRING) {
            kendo.culture(culture);
            culture = kendo.culture();
        }

        if (!formats) {
            formats = [];
            patterns = culture.calendar.patterns;
            length = formatsSequence.length;

            for (; idx < length; idx++) {
                formats[idx] = patterns[formatsSequence[idx]];
            }
            formats[idx] = "ddd MMM dd yyyy HH:mm:ss";

            idx = 0;
        }

        formats = isArray(formats) ? formats: [formats];
        length = formats.length;

        for (; idx < length; idx++) {
            date = parseExact(value, formats[idx], culture);
            if (date) {
                return date;
            }
        }

        return date;
    };

    kendo.parseInt = function(value, culture) {
        var result = kendo.parseFloat(value, culture);
        if (result) {
            result = result | 0;
        }
        return result;
    };

    kendo.parseFloat = function(value, culture) {
        if (!value && value !== 0) {
           return null;
        }

        if (typeof value === NUMBER) {
           return value;
        }

        value = value.toString();
        culture = kendo.cultures[culture] || kendo.cultures.current;

        var number = culture.numberFormat,
            percent = number.percent,
            currency = number.currency,
            symbol = currency.symbol,
            percentSymbol = percent.symbol,
            negative = value.indexOf("-") > -1,
            parts;

        //handle exponential number
        if (exponentRegExp.test(value)) {
            value = parseFloat(value);
            if (isNaN(value)) {
                value = null;
            }
            return value;
        }

        if (value.indexOf(symbol) > -1) {
            number = currency;
            parts = number.pattern[0].replace("$", symbol).split("n");
            if (value.indexOf(parts[0]) > -1 && value.indexOf(parts[1]) > -1) {
                value = value.replace(parts[0], "").replace(parts[1], "");
                negative = true;
            }
        } else if (value.indexOf(percentSymbol) > -1) {
            number = percent;
            symbol = percentSymbol;
        }

        value = value.replace("-", "")
                     .replace(symbol, "")
                     .replace(nonBreakingSpaceRegExp, " ")
                     .split(number[","].replace(nonBreakingSpaceRegExp, " ")).join("")
                     .replace(number["."], ".");

        value = parseFloat(value);

        if (isNaN(value)) {
            value = null;
        } else if (negative) {
            value *= -1;
        }

        return value;
    };

    if (globalize) {
        kendo.parseDate = proxy(globalize.parseDate, globalize);
        kendo.parseFloat = proxy(globalize.parseFloat, globalize);
    }
})();

    function wrap(element) {
        var browser = $.browser;

        if (!element.parent().hasClass("k-animation-container")) {
            var shadow = element.css(kendo.support.transitions.css + "box-shadow") || element.css("box-shadow"),
                radius = shadow ? shadow.match(boxShadowRegExp) || [ 0, 0, 0, 0, 0 ] : [ 0, 0, 0, 0, 0 ],
                blur = math.max((+radius[3]), +(radius[4] || 0)),
                left = (-radius[1]) + blur,
                right = (+radius[1]) + blur,
                bottom = (+radius[2]) + blur;

            if (browser.opera) { // Box shadow can't be retrieved in Opera
                left = right = bottom = 5;
            }

            element.wrap(
                         $("<div/>")
                         .addClass("k-animation-container")
                         .css({
                             width: element.outerWidth(),
                             height: element.outerHeight(),
                             marginLeft: -left,
                             paddingLeft: left,
                             paddingRight: right,
                             paddingBottom: bottom
                         }));
        } else {
            var wrap = element.parent(".k-animation-container");

            if (wrap.is(":hidden")) {
                wrap.show();
            }

            wrap.css({
                    width: element.outerWidth(),
                    height: element.outerHeight()
                });
        }

        if (browser.msie && math.floor(browser.version) <= 7) {
            element.css({
                zoom: 1
            });
        }

        return element.parent();
    }

    function deepExtend(destination) {
        var i = 1,
            length = arguments.length;

        for (i = 1; i < length; i++) {
            deepExtendOne(destination, arguments[i]);
        }

        return destination;
    }

    function deepExtendOne(destination, source) {
        var property,
            propValue,
            propType,
            destProp;

        for (property in source) {
            propValue = source[property];
            propType = typeof propValue;
            if (propType === OBJECT && propValue !== null && propValue.constructor !== Array) {
                destProp = destination[property];
                if (typeof (destProp) === OBJECT) {
                    destination[property] = destProp || {};
                } else {
                    destination[property] = {};
                }
                deepExtendOne(destination[property], propValue);
            } else if (propType !== UNDEFINED) {
                destination[property] = propValue;
            }
        }

        return destination;
    }

    /**
     * Contains results from feature detection.
     * @name kendo.support
     * @namespace Contains results from feature detection.
     */
    (function() {
        /**
         * Indicates the width of the browser scrollbar. A value of zero means that the browser does not show a visual representation of a scrollbar (i.e. mobile browsers).
         * @name kendo.support.scrollbar
         * @property {Boolean}
         */
        support.scrollbar = function() {
            var div = document.createElement("div"),
                result;

            div.style.cssText = "overflow:scroll;overflow-x:hidden;zoom:1;clear:both";
            div.innerHTML = "&nbsp;";
            document.body.appendChild(div);

            result = div.offsetWidth - div.scrollWidth;

            document.body.removeChild(div);
            return result;
        };

        var table = document.createElement("table");

        // Internet Explorer does not support setting the innerHTML of TBODY and TABLE elements
        try {
            table.innerHTML = "<tr><td></td></tr>";

            /**
             * Indicates whether the browser supports setting of the &lt;tbody&gt; innerHtml.
             * @name kendo.support.tbodyInnerHtml
             * @property {Boolean}
             */
            support.tbodyInnerHtml = true;
        } catch (e) {
            support.tbodyInnerHtml = false;
        }

        /**
         * Indicates whether the browser supports touch events.
         * @name kendo.support.touch
         * @property {Boolean}
         */
        support.touch = "ontouchstart" in window;
        support.pointers = navigator.msPointerEnabled;

        /**
         * Indicates whether the browser supports CSS transitions.
         * @name kendo.support.transitions
         * @property {Boolean}
         */
        var transitions = support.transitions = false;
        var transforms = support.transforms = false;

        /**
         * Indicates whether the browser supports hardware 3d transitions.
         * @name kendo.support.hasHW3D
         * @property {Boolean}
         */
        support.hasHW3D = ("WebKitCSSMatrix" in window && "m11" in new WebKitCSSMatrix()) || "MozPerspective" in document.documentElement.style;
        support.hasNativeScrolling = typeof document.documentElement.style.webkitOverflowScrolling == "string";

        each([ "Moz", "webkit", "O", "ms" ], function () {
            var prefix = this.toString(),
                hasTransitions = typeof table.style[prefix + "Transition"] === STRING;

            if (hasTransitions || typeof table.style[prefix + "Transform"] === STRING) {
                var lowPrefix = prefix.toLowerCase();

                transforms = {
                    css: "-" + lowPrefix + "-",
                    prefix: prefix,
                    event: (lowPrefix === "o" || lowPrefix === "webkit") ? lowPrefix : ""
                };

                if (hasTransitions) {
                    transitions = transforms;
                    transitions.event = transitions.event ? transitions.event + "TransitionEnd" : "transitionend";
                }

                return false;
            }
        });

        support.transforms = transforms;
        support.transitions = transitions;

        support.detectOS = function (ua) {
            var os = false, match = [],
                agentRxs = {
                    fire: /(Silk)\/(\d+)\.(\d+(\.\d+)?)/,
                    android: /(Android)\s+(\d+)\.(\d+(\.\d+)?)/,
                    iphone: /(iPhone|iPod).*OS\s+(\d+)[\._]([\d\._]+)/,
                    ipad: /(iPad).*OS\s+(\d+)[\._]([\d_]+)/,
                    meego: /(MeeGo).+NokiaBrowser\/(\d+)\.([\d\._]+)/,
                    webos: /(webOS)\/(\d+)\.(\d+(\.\d+)?)/,
                    blackberry: /(BlackBerry|PlayBook).*?Version\/(\d+)\.(\d+(\.\d+)?)/
                },
                osRxs = {
                    ios: /^i(phone|pad|pod)$/i,
                    android: /^android|fire$/i
                },
                testOs = function (agent) {
                    for (var os in osRxs) {
                        if (osRxs.hasOwnProperty(os) && osRxs[os].test(agent))
                            return os;
                    }
                    return agent;
                };

            for (var agent in agentRxs) {
                if (agentRxs.hasOwnProperty(agent)) {
                    match = ua.match(agentRxs[agent]);
                    if (match) {
                        os = {};
                        os.device = agent;
                        os.name = testOs(agent);
                        os[os.name] = true;
                        os.majorVersion = match[2];
                        os.minorVersion = match[3].replace("_", ".");
                        os.flatVersion = os.majorVersion + os.minorVersion.replace(".", "");
                        os.flatVersion = os.flatVersion + (new Array(4 - os.flatVersion.length).join("0")); // Pad with zeroes
                        os.appMode = window.navigator.standalone || (/file|local/).test(window.location.protocol) || typeof window.PhoneGap !== UNDEFINED; // Use file protocol to detect appModes.

                        break;
                    }
                }
            }
            return os;
        };

        /**
         * Parses the mobile OS type and version from the browser user agent.
         * @name kendo.support.mobileOS
         */
        support.mobileOS = support.detectOS(navigator.userAgent);

        support.zoomLevel = function() {
            return support.touch ? (document.documentElement.clientWidth / window.innerWidth) : 1;
        };

        /**
         * Indicates the browser device pixel ratio.
         * @name kendo.support.devicePixelRatio
         * @property {Float}
         */
        support.devicePixelRatio = window.devicePixelRatio === undefined ? 1 : window.devicePixelRatio;

        /**
         * Indicates whether the browser supports input placeholder.
         * @name kendo.support.placeholder
         * @property {Boolean}
         */
        support.placeholder = "placeholder" in document.createElement("input");
        support.stableSort = (function() {
            var sorted = [0,1,2,3,4,5,6,7,8,9,10,11,12].sort(function() { return 0 } );
            return sorted[0] === 0 && sorted[1] === 1 && sorted[2] === 2 && sorted[3] === 3 && sorted[4] === 4 &&
                sorted[5] === 5 && sorted[6] === 6 && sorted[7] === 7 && sorted[8] === 8 &&
                sorted[9] === 9 && sorted[10] === 10 && sorted[11] === 11 && sorted[12] === 12;
        })();

    })();

    /**
     * Exposed by jQuery.
     * @ignore
     * @name jQuery.fn
     * @namespace Handy jQuery plug-ins that are used by all Kendo widgets.
     */

    function size(obj) {
        var size = 0, key;
        for (key in obj) {
            obj.hasOwnProperty(key) && size++;
        }

        return size;
    }

    function isNodeEmpty(element) {
        return $.trim($(element).contents().filter(function () { return this.nodeType != 8 }).html()) === "";
    }

    function getOffset(element, type) {
        if (!type) {
            type = "offset";
        }

        var result = element[type](),
            mobileOS = support.mobileOS;

        if (support.touch && mobileOS.ios && mobileOS.flatVersion < 410) { // Extra processing only in broken iOS'
            var offset = type == "offset" ? result : element.offset(),
                positioned = (result.left == offset.left && result.top == offset.top);

            if (positioned) {
                return {
                    top: result.top - window.scrollY,
                    left: result.left - window.scrollX
                };
            }
        }

        return result;
    }

    var directions = {
        left: { reverse: "right" },
        right: { reverse: "left" },
        down: { reverse: "up" },
        up: { reverse: "down" },
        top: { reverse: "bottom" },
        bottom: { reverse: "top" },
        "in": { reverse: "out" },
        out: { reverse: "in" }
    };

    function parseEffects(input) {
        var effects = {};

        each((typeof input === "string" ? input.split(" ") : input), function(idx) {
            effects[idx] = this;
        });

        return effects;
    }

    var fx = {
        promise: function (element, options) {
            if (options.show) {
                element.css({ display: element.data("olddisplay") || "block" }).css("display");
            }

            if (options.hide) {
                element.data("olddisplay", element.css("display")).hide();
            }

            if (options.init) {
                options.init();
            }

            if (options.completeCallback) {
                options.completeCallback(element); // call the external complete callback with the element
            }

            element.dequeue();
        },

        transitionPromise: function(element, destination, options) {
            var container = kendo.wrap(element);
            container.append(destination);

            element.hide();
            destination.show();

            if (options.completeCallback) {
                options.completeCallback(element); // call the external complete callback with the element
            }

            return element;
        }
    };

    function prepareAnimationOptions(options, duration, reverse, complete) {
        if (typeof options === STRING) {
            // options is the list of effect names separated by space e.g. animate(element, "fadeIn slideDown")

            // only callback is provided e.g. animate(element, options, function() {});
            if (isFunction(duration)) {
                complete = duration;
                duration = 400;
                reverse = false;
            }

            if (isFunction(reverse)) {
                complete = reverse;
                reverse = false;
            }

            if (typeof duration === BOOLEAN){
                reverse = duration;
                duration = 400;
            }

            options = {
                effects: options,
                duration: duration,
                reverse: reverse,
                complete: complete
            };
        }

        return extend({
            //default options
            effects: {},
            duration: 400, //jQuery default duration
            reverse: false,
            init: noop,
            teardown: noop,
            hide: false,
            show: false
        }, options, { completeCallback: options.complete, complete: noop }); // Move external complete callback, so deferred.resolve can be always executed.

    }

    function animate(element, options, duration, reverse, complete) {
        element.each(function (idx, el) { // fire separate queues on every element to separate the callback elements
            el = $(el);
            el.queue(function () {
                fx.promise(el, prepareAnimationOptions(options, duration, reverse, complete));
            });
        });

        return element;
    }

    function animateTo(element, destination, options, duration, reverse, complete) {
        return fx.transitionPromise(element, destination, prepareAnimationOptions(options, duration, reverse, complete));
    }

    extend($.fn, /** @lends jQuery.fn */{
        kendoStop: function(clearQueue, gotoEnd) {
            return this.stop(clearQueue, gotoEnd);
        },

        kendoAnimate: function(options, duration, reverse, complete) {
            return animate(this, options, duration, reverse, complete);
        },

        kendoAnimateTo: function(destination, options, duration, reverse, complete) {
            return animateTo(this, destination, options, duration, reverse, complete);
        }
    });

    function toggleClass(element, classes, options, add) {
        if (classes) {
            classes = classes.split(" ");

            each(classes, function(idx, value) {
                element.toggleClass(value, add);
            });
        }

        return element;
    }

    extend($.fn, /** @lends jQuery.fn */{
        kendoAddClass: function(classes, options){
            return toggleClass(this, classes, options, true);
        },
        kendoRemoveClass: function(classes, options){
            return toggleClass(this, classes, options, false);
        },
        kendoToggleClass: function(classes, options, toggle){
            return toggleClass(this, classes, options, toggle);
        }
    });

    var ampRegExp = /&/g,
        ltRegExp = /</g,
        gtRegExp = />/g;
    /**
     * Encodes HTML characters to entities.
     * @name kendo.htmlEncode
     * @function
     * @param {String} value The string that needs to be HTML encoded.
     * @returns {String} The encoded string.
     */
    function htmlEncode(value) {
        return ("" + value).replace(ampRegExp, "&amp;").replace(ltRegExp, "&lt;").replace(gtRegExp, "&gt;");
    }

    var touchLocation = function(e) {
        return {
            idx: 0,
            x: e.pageX,
            y: e.pageY
        };
    };

    var eventTarget = function (e) {
        return e.target;
    };

    if (support.touch) {
        /** @ignore */
        touchLocation = function(e, id) {
            var changedTouches = e.changedTouches || e.originalEvent.changedTouches;

            if (id) {
                var output = null;
                each(changedTouches, function(idx, value) {
                    if (id == value.identifier) {
                        output = {
                            idx: value.identifier,
                            x: value.pageX,
                            y: value.pageY
                        };
                    }
                });
                return output;
            } else {
                return {
                    idx: changedTouches[0].identifier,
                    x: changedTouches[0].pageX,
                    y: changedTouches[0].pageY
                };
            }
        };

        eventTarget = function(e) {
            var touches = "originalEvent" in e ? e.originalEvent.changedTouches : "changedTouches" in e ? e.changedTouches : null;

            return touches ? document.elementFromPoint(touches[0].clientX, touches[0].clientY) : null;
        };

        each(["swipe", "swipeLeft", "swipeRight", "swipeUp", "swipeDown", "doubleTap", "tap"], function(m, value) {
            $.fn[value] = function(callback) {
                return this.bind(value, callback)
            }
        });
    }

    if (support.touch) {
        support.mousedown = "touchstart";
        support.mouseup = "touchend";
        support.mousemove = "touchmove";
        support.mousecancel = "touchcancel";
    } else {
        support.mousemove = "mousemove";
        support.mousedown = "mousedown";
        support.mouseup = "mouseup";
        support.mousecancel = "mouseleave";
    }

    var wrapExpression = function(members) {
        var result = "d",
            index,
            idx,
            length,
            member,
            count = 1;

        for (idx = 0, length = members.length; idx < length; idx++) {
            member = members[idx];
            if (member !== "") {
                index = member.indexOf("[");

                if (index != 0) {
                    if (index == -1) {
                        member = "." + member;
                    } else {
                        count++;
                        member = "." + member.substring(0, index) + " || {})" + member.substring(index);
                    }
                }

                count++;
                result += member + ((idx < length - 1) ? " || {})" : ")");
            }
        }
        return new Array(count).join("(") + result;
    },
    localUrlRe = /^([a-z]+:)?\/\//i;

    extend(kendo, /** @lends kendo */ {
        /**
         * @name kendo.ui
         * @namespace Contains all classes for the Kendo UI widgets.
         */
        ui: {
            /**
             * Shows an overlay with a loading message, indicating that an action is in progress.
             * @name kendo.ui.progress
             * @function
             * @param {jQueryObject} container The container that will hold the overlay
             * @param {Boolean} toggle Whether the overlay should be shown or hidden
             */
            progress: function(container, toggle) {
                var mask = container.find(".k-loading-mask");

                if (toggle) {
                    if (!mask.length) {
                        mask = $("<div class='k-loading-mask'><span class='k-loading-text'>Loading...</span><div class='k-loading-image'/><div class='k-loading-color'/></div>")
                            .width("100%").height("100%")
                            .prependTo(container)
                            .css({ top: container.scrollTop(), left: container.scrollLeft() });
                    }
                } else if (mask) {
                    mask.remove();
                }
            }
        },
        fx: fx,
        data: {},
        mobile: {},
        keys: {
            DELETE: 46,
            BACKSPACE: 8,
            TAB: 9,
            ENTER: 13,
            ESC: 27,
            LEFT: 37,
            UP: 38,
            RIGHT: 39,
            DOWN: 40,
            END: 35,
            HOME: 36,
            SPACEBAR: 32,
            PAGEUP: 33,
            PAGEDOWN: 34,
            F10: 121,
            F12: 123
        },
        support: support,
        animate: animate,
        ns: "",
        attr: function(value) {
            return "data-" + kendo.ns + value;
        },
        wrap: wrap,
        deepExtend: deepExtend,
        size: size,
        isNodeEmpty: isNodeEmpty,
        getOffset: getOffset,
        parseEffects: parseEffects,
        toggleClass: toggleClass,
        directions: directions,
        Observable: Observable,
        Class: Class,
        Template: Template,
        /**
         * Shorthand for {@link kendo.Template.compile}.
         * @name kendo.template
         * @function
         */
        template: proxy(Template.compile, Template),
        /**
         * Shorthand for {@link kendo.Template.render}.
         * @name kendo.render
         * @function
         */
        render: proxy(Template.render, Template),
        stringify: proxy(JSON.stringify, JSON),
        touchLocation: touchLocation,
        eventTarget: eventTarget,
        htmlEncode: htmlEncode,
        isLocalUrl: function(url) {
            return url && !localUrlRe.test(url);
        },
        /** @ignore */
        expr: function(expression, safe) {
            expression = expression || "";

            if (expression && expression.charAt(0) !== "[") {
                expression = "." + expression;
            }

            if (safe) {
                expression =  wrapExpression(expression.split("."));
            } else {
                expression = "d" + expression;
            }

            return expression;
        },
        /** @ignore */
        getter: function(expression, safe) {
            return getterCache[expression] = getterCache[expression] || new Function("d", "return " + kendo.expr(expression, safe));
        },
        /** @ignore */
        setter: function(expression) {
            return setterCache[expression] = setterCache[expression] || new Function("d,value", "d." + expression + "=value");
        },
        /** @ignore */
        accessor: function(expression) {
            return {
                get: kendo.getter(expression),
                set: kendo.setter(expression)
            };
        },
        /** @ignore */
        guid: function() {
            var id = "", i, random;

            for (i = 0; i < 32; i++) {
                random = math.random() * 16 | 0;

                if (i == 8 || i == 12 || i == 16 || i == 20) {
                    id += "-";
                }
                id += (i == 12 ? 4 : (i == 16 ? (random & 3 | 8) : random)).toString(16);
            }

            return id;
        },

        roleSelector: function(role) {
            return "[" + kendo.attr("role") + "=" + role + "]";
        },

        /** @ignore */
        logToConsole: function(message) {
            if (typeof(console) != "undefined" && console.log) {
                console.log(message);
            }
        }
    });

    var Widget = Observable.extend( /** @lends kendo.ui.Widget.prototype */ {
        /**
         * Initializes widget. Sets `element` and `options` properties.
         * @constructs
         * @class Represents a UI widget. Base class for all Kendo widgets
         * @extends kendo.Observable
         */
        init: function(element, options) {
            var that = this;

            that.element = $(element);

            Observable.fn.init.call(that);

            that.options = extend(true, {}, that.options, options);

            if (!that.element.attr(kendo.attr("role"))) {
                that.element.attr(kendo.attr("role"), (that.options.name || "").toLowerCase());
            }

            that.element.data("kendo" + that.options.prefix + that.options.name, that);

            that.bind(that.events, that.options);
        },

        events: [],

        options: {
            prefix: ""
        },

        setOptions: function(options) {
            $.extend(this.options, options);

            this.bind(this.events, options);
        }
    });

    kendo.notify = noop;

    var templateRegExp = /template$/i,
        jsonRegExp = /^(?:\{.*\}|\[.*\])$/,
        dashRegExp = /([A-Z])/g;

    function parseOption(element, option) {
        var value;

        if (option.indexOf("data") === 0) {
            option = option.substring(4);
            option = option.charAt(0).toLowerCase() + option.substring(1);
        }

        option = option.replace(dashRegExp, "-$1");
        value = element.getAttribute("data-" + kendo.ns + option);

        if (value === null) {
            value = undefined;
        } else if (value === "null") {
            value = null;
        } else if (value === "true") {
            value = true;
        } else if (value === "false") {
            value = false;
        } else if (!isNaN(parseFloat(value))) {
            value = parseFloat(value);
        } else if (jsonRegExp.test(value)) {
           value = $.parseJSON(value);
        }

        return value;
    }

    function parseOptions(element, options) {
        var result = {},
            value;

        for (option in options) {
            value = parseOption(element, option);

            if (value !== undefined) {

                if (templateRegExp.test(option)) {
                    value = kendo.template($("#" + value).html());
                }

                result[option] = value;
            }
        }

        return result;
    }

    kendo.initWidget = function(element, options, namespace) {
        var result,
            option,
            widget,
            idx,
            length,
            role,
            dataSource;

        element = element.nodeType ? element : element[0];

        role = element.getAttribute("data-" + kendo.ns + "role");

        if (!role) {
            return;
        }

        dataSource = parseOption(element, "dataSource");

        widget = (namespace || kendo.ui).roles[role];

        if (!widget) {
            return;
        }

        options = $.extend({}, parseOptions(element, widget.fn.options), options);

        if (dataSource) {
            if (typeof dataSource === STRING) {
                options.dataSource = kendo.getter(dataSource)(window);
            } else {
                options.dataSource = dataSource;
            }
        }

        for (idx = 0, length = widget.fn.events.length; idx < length; idx++) {
            option = widget.fn.events[idx];

            value = parseOption(element, option);

            if (value !== undefined) {
                options[option] = kendo.getter(value)(window);
            }
        }

        result = $(element).data("kendo" + widget.fn.options.name);

        if (!result) {
            result = new widget(element, options);
        } else {
            result.setOptions(options);
        }

        return result;
    }

    kendo.init = function(element, namespace) {
        $(element).find("[data-" + kendo.ns + "role]").andSelf().each(function(){
            kendo.initWidget(this, {}, namespace);
        });
    }

    kendo.parseOptions = parseOptions;

    extend(kendo.ui, /** @lends kendo.ui */{
        Widget: Widget,
        roles: {},
        /**
         * Helper method for writing new widgets.
         * Exposes a jQuery plug-in that will handle the widget creation and attach its client-side object in the appropriate data-* attribute.
         * @name kendo.ui.plugin
         * @function
         * @param {kendo.ui.Widget} widget The widget function.
         * @param {Object} register <kendo.ui> The object where the reference to the widget is recorded.
         * @param {Object} prefix <""> The plugin function prefix, e.g. "Mobile" will register "kendoMobileFoo".
         * @example
         * function TextBox(element, options);
         * kendo.ui.plugin(TextBox);
         *
         * // initialize a new TextBox for each input, with the given options object.
         * $("input").kendoTextBox({ });
         * // get the TextBox object and call the value API method
         * $("input").data("kendoTextBox").value();
         */
        plugin: function(widget, register, prefix) {
            var name = widget.fn.options.name;

            register = register || kendo.ui;
            prefix = prefix || "";

            register[name] = widget;

            register.roles[name.toLowerCase()] = widget;

            name = "kendo" + prefix + name;
            // expose a jQuery plugin
            $.fn[name] = function(options) {
                $(this).each(function() {
                    new widget(this, options);
                });
                return this;
            }
        }
    });

    var MobileWidget = Widget.extend(/** @lends kendo.mobile.ui.Widget.prototype */{
        /**
         * Initializes mobile widget. Sets `element` and `options` properties.
         * @constructs
         * @class Represents a mobile UI widget. Base class for all Kendo mobile widgets.
         * @extends kendo.ui.Widget
         */
        init: function(element, options) {
            Widget.fn.init.call(this, element, options);
            this.wrapper = this.element;
        },

        options: {
            prefix: "Mobile"
        },

        events: [],

        viewShow: $.noop,

        viewInit: function(view) {
            this.view = view;
        }
    });

    /**
     * @name kendo.mobile
     * @namespace This object contains all code introduced by the Kendo mobile suite, plus helper functions that are used across all mobile widgets.
     */
    $.extend(kendo.mobile, {
        init: function(element) {
            kendo.init(element, kendo.mobile.ui);
        },

        /**
         * @name kendo.mobile.ui
         * @namespace Contains all classes for the Kendo Mobile UI widgets.
         */
        ui: {
            Widget: MobileWidget,
            roles: {},
            plugin: function(widget) {
                kendo.ui.plugin(widget, kendo.mobile.ui, "Mobile");
            }
        }
    });

    /**
     * Enables kinetic scrolling on touch devices
     * @name kendo.touchScroller
     * @function
     * @param {Selector} element The container element to enable scrolling for.
     */
    kendo.touchScroller = function(element, options) {
        if (support.touch && kendo.mobile.ui.Scroller) {
            element.kendoMobileScroller(options);
            return element.data("kendoMobileScroller");
        } else {
            return false;
        }
    }
})(jQuery);
/*
* Kendo UI Web v2012.1.322 (http://kendoui.com)
* Copyright 2012 Telerik AD. All rights reserved.
*
* Kendo UI Web commercial licenses may be obtained at http://kendoui.com/web-license
* If you do not own a commercial license, this file shall be governed by the
* GNU General Public License (GPL) version 3.
* For GPL requirements, please review: http://www.gnu.org/copyleft/gpl.html
*/
(function($, undefined) {
    var extend = $.extend,
        proxy = $.proxy,
        isFunction = $.isFunction,
        isPlainObject = $.isPlainObject,
        isEmptyObject = $.isEmptyObject,
        isArray = $.isArray,
        grep = $.grep,
        ajax = $.ajax,
        map,
        each = $.each,
        noop = $.noop,
        kendo = window.kendo,
        Observable = kendo.Observable,
        Class = kendo.Class,
        STRING = "string",
        FUNCTION = "function",
        CREATE = "create",
        READ = "read",
        UPDATE = "update",
        DESTROY = "destroy",
        CHANGE = "change",
        GET = "get",
        MULTIPLE = "multiple",
        SINGLE = "single",
        ERROR = "error",
        REQUESTSTART = "requestStart",
        crud = [CREATE, READ, UPDATE, DESTROY],
        identity = function(o) { return o; },
        getter = kendo.getter,
        stringify = kendo.stringify,
        math = Math,
        push = [].push,
        join = [].join,
        pop = [].pop,
        splice = [].splice,
        shift = [].shift,
        slice = [].slice,
        unshift = [].unshift,
        toString = {}.toString,
        stableSort = kendo.support.stableSort,
        dateRegExp = /^\/Date\((.*?)\)\/$/,
        quoteRegExp = /(?=['\\])/g;

    var ObservableArray = Observable.extend({
        init: function(array, type) {
            var that = this;

            that.type = type || ObservableObject;

            Observable.fn.init.call(that);

            that.length = array.length;

            that.wrapAll(array, that);
        },

        wrapAll: function(source, target) {
            var idx, length;

            target = target || [];

            for (idx = 0, length = source.length; idx < length; idx++) {
                target[idx] = this.wrap(source[idx]);
            }

            return target;
        },

        wrap: function(object) {
            var that = this;

            if (object !== null && toString.call(object) === "[object Object]") {
                var observable = object instanceof that.type || object instanceof Model;
                if (!observable) {
                    object = object instanceof ObservableObject ? object.toJSON() : object;
                    object = new that.type(object);
                }

                object.bind(CHANGE, function(e) {
                    that.trigger(CHANGE, {
                        field: e.field,
                        items: [this],
                        action: "itemchange"
                    });
                });
            }

            return object;
        },

        push: function() {
            var index = this.length,
                items = this.wrapAll(arguments),
                result;

            result = push.apply(this, items);

            this.trigger(CHANGE, {
                action: "add",
                index: index,
                items: items
            });

            return result;
        },

        slice: slice,

        join: join,

        pop: function() {
            var length = this.length, result = pop.apply(this);

            if (length) {
                this.trigger(CHANGE, {
                    action: "remove",
                    index: length - 1,
                    items:[result]
                });
            }

            return result;
        },

        splice: function(index, howMany, item) {
            var items = this.wrapAll(slice.call(arguments, 2)),
                result;

            result = splice.apply(this, [index, howMany].concat(items));

            if (result.length) {
                this.trigger(CHANGE, {
                    action: "remove",
                    index: index,
                    items: result
                });
            }

            if (item) {
                this.trigger(CHANGE, {
                    action: "add",
                    index: index,
                    items: items
                });
            }
            return result;
        },

        shift: function() {
            var length = this.length, result = shift.apply(this);

            if (length) {
                this.trigger(CHANGE, {
                    action: "remove",
                    index: 0,
                    items:[result]
                });
            }

            return result;
        },

        unshift: function() {
            var items = this.wrapAll(arguments),
                result;

            result = unshift.apply(this, items);

            this.trigger(CHANGE, {
                action: "add",
                index: 0,
                items: items
            });

            return result;
        },

        indexOf: function(item) {
            var that = this,
                idx,
                length;

            for (idx = 0, length = that.length; idx < length; idx++) {
                if (that[idx] === item) {
                    return idx;
                }
            }
            return -1;
        }
    });

    var ObservableObject = Observable.extend({
        init: function(value) {
            var that = this,
                member,
                field,
                type;

            Observable.fn.init.call(this);

            for (field in value) {
                member = value[field];
                if (field.charAt(0) != "_") {
                    type = toString.call(member);

                    member = that.wrap(member, field);
                }
                that[field] = member;
            }

            that.uid = kendo.guid();
        },

        shouldSerialize: function(field) {
            return this.hasOwnProperty(field) && field !== "_events" && typeof this[field] !== FUNCTION
            && field !== "uid";
        },

        toJSON: function() {
            var result = {}, field;

            for (field in this) {
                if (this.shouldSerialize(field)) {
                    result[field] = this[field];
                }
            }

            return result;
        },

        get: function(field, call) {
            var that = this, result, getter;

            that.trigger(GET, { field: field });

            if (field === "this") {
                result = that;
            } else {
                getter = kendo.getter(field, true);

                result = getter(that);

                if (call && typeof result === FUNCTION) {
                    result = result.call(that);
                }
            }

            return result;
        },

        _set: function(field, value) {
            var that = this;
            if (field.indexOf(".")) {
                var paths = field.split("."),
                    path = "";

                while (paths.length > 1) {
                    path += paths.shift();
                    var obj = kendo.getter(path, true)(that);
                    if (obj instanceof ObservableObject) {
                        obj.set(paths.join("."), value);
                        return;
                    }
                    path += ".";
                }
            }

            kendo.setter(field)(that, value);
        },

        set: function(field, value) {
            var that = this,
                current = that[field],
                setter;

            if (current != value) {
                if (!that.trigger("set", { field: field, value: value })) {

                    that._set(field, that.wrap(value, field));

                    that.trigger(CHANGE, { field: field });
                }
            }
        },

        wrap: function(object, field) {
            var that = this,
                type = toString.call(object);

            if (object !== null && type === "[object Object]" && !(object instanceof DataSource)) {
                if (!(object instanceof ObservableObject)) {
                    object = new ObservableObject(object);
                }

                (function(field) {
                    object.bind(GET, function(e) {
                        e.field = field + "." + e.field;
                        that.trigger(GET, e);
                    });

                    object.bind(CHANGE, function(e) {
                        e.field = field + "." + e.field;
                        that.trigger(CHANGE, e);
                    });
                })(field);
            } else if (object !== null && type === "[object Array]") {
                object = new ObservableArray(object);

                (function(field) {
                    object.bind(CHANGE, function(e) {
                        that.trigger(CHANGE, { field: field, index: e.index, items: e.items, action: e.action});
                    });
                })(field);
            }
            return object;
        }
    });

    function equal(x, y) {
        if (x === y) {
            return true;
        }

        var xtype = $.type(x), ytype = $.type(y), field;

        if (xtype !== ytype) {
            return false;
        }

        if (xtype === "date") {
            return x.getTime() === y.getTime();
        }

        if (xtype !== "object" && xtype !== "array") {
            return false;
        }

        for (field in x) {
            if (!equal(x[field], y[field])) {
                return false;
            }
        }

        return true;
    }

    var parsers = {
        "number": function(value) {
            return kendo.parseFloat(value);
        },

        "date": function(value) {
            if (typeof value === STRING) {
                var date = dateRegExp.exec(value);
                if (date) {
                    return new Date(parseInt(date[1]));
                }
            }
            return kendo.parseDate(value);
        },

        "boolean": function(value) {
            if (typeof value === STRING) {
                return value.toLowerCase() === "true";
            }
            return !!value;
        },

        "string": function(value) {
            return value + "";
        },

        "default": function(value) {
            return value;
        }
    };

    var defaultValues = {
        "string": "",
        "number": 0,
        "date": new Date(),
        "boolean": false,
        "default": ""
    }

    var Model = ObservableObject.extend({
        init: function(data) {
            var that = this;

            if (!data || $.isEmptyObject(data)) {
                data = $.extend({}, that.defaults, data);
            }

            ObservableObject.fn.init.call(that, data);

            that.dirty = false;

            if (that.idField) {
                that.id = that.get(that.idField);

                if (that.id === undefined) {
                    that.id = that._defaultId;
                }
            }
        },

        shouldSerialize: function(field) {
            return ObservableObject.fn.shouldSerialize.call(this, field)
            && field !== "uid"
            && !(this.idField !== "id" && field === "id")
            && field !== "dirty" && field !== "_accessors";
        },

        _parse: function(field, value) {
            var that = this,
            parse;

            field = (that.fields || {})[field];
            if (field) {
                parse = field.parse;
                if (!parse && field.type) {
                    parse = parsers[field.type.toLowerCase()];
                }
            }

            return parse ? parse(value) : value;
        },

        editable: function(field) {
            field = (this.fields || {})[field];
            return field ? field.editable !== false : true;
        },

        set: function(field, value, initiator) {
            var that = this;

            if (that.editable(field)) {
                value = that._parse(field, value);

                if (!equal(value, that.get(field))) {
                    ObservableObject.fn.set.call(that, field, value, initiator);
                    that.dirty = true;
                }
            }
        },

        accept: function(data) {
            var that = this;

            extend(that, data);

            if (that.idField) {
                that.id = that.get(that.idField);
            }
            that.dirty = false;
        },

        isNew: function() {
            return this.id === this._defaultId;
        }
    });

    Model.define = function(options) {
        var model,
            proto = extend({}, { defaults: {} }, options),
            id = proto.id;

        if (id) {
            proto.idField = id;
        }

        if (proto.id) {
            delete proto.id;
        }

        if (id) {
            proto.defaults[id] = proto._defaultId = "";
        }

        for (var name in proto.fields) {
            var field = proto.fields[name],
            type = field.type || "default",
            value = null;

            name = field.field || name;

            if (!field.nullable) {
                value = proto.defaults[name] = field.defaultValue !== undefined ? field.defaultValue : defaultValues[type.toLowerCase()];
            }

            if (options.id === name) {
                proto._defaultId = value;
            }

            proto.defaults[name] = value;

            field.parse = field.parse || parsers[type];
        }

        model = Model.extend(proto);

        if (proto.fields) {
            model.fields = proto.fields;
            model.idField = proto.idField;
        }

        return model;
    }

    var Comparer = {
        selector: function(field) {
            return isFunction(field) ? field : getter(field);
        },

        asc: function(field) {
            var selector = this.selector(field);
            return function (a, b) {
                a = selector(a);
                b = selector(b);

                return a > b ? 1 : (a < b ? -1 : 0);
            };
        },

        desc: function(field) {
            var selector = this.selector(field);
            return function (a, b) {
                a = selector(a);
                b = selector(b);

                return a < b ? 1 : (a > b ? -1 : 0);
            };
        },

        create: function(descriptor) {
            return Comparer[descriptor.dir.toLowerCase()](descriptor.field);
        },

        combine: function(comparers) {
            return function(a, b) {
                var result = comparers[0](a, b),
                idx,
                length;

                for (idx = 1, length = comparers.length; idx < length; idx ++) {
                    result = result || comparers[idx](a, b);
                }

                return result;
            }
        }
    };

    var PositionComparer = {
        selector: function(field) {
            return isFunction(field) ? field : getter(field);
        },

        asc: function(field) {
            var selector = this.selector(field);
            return function (a, b) {
                var valueA = selector(a);
                var valueB = selector(b);

                if (valueA === valueB) {
                    return a.__position - b.__position;
                }
                return valueA > valueB ? 1 : (valueA < valueB ? -1 : 0);
            };
        },

        desc: function(field) {
            var selector = this.selector(field);
            return function (a, b) {
                var valueA = selector(a);
                var valueB = selector(b);

                if (valueA === valueB) {
                    return a.__position - b.__position;
                }

                return valueA < valueB ? 1 : (valueA > valueB ? -1 : 0);
            };
        },

        create: function(descriptor) {
            return PositionComparer[descriptor.dir.toLowerCase()](descriptor.field);
        },

        combine: function(comparers) {
             return function(a, b) {
                 var result = comparers[0](a, b),
                     idx,
                     length;

                 for (idx = 1, length = comparers.length; idx < length; idx ++) {
                     result = result || comparers[idx](a, b);
                 }

                 return result;
             }
        }
    };

    map = function (array, callback) {
        var idx, length = array.length, result = new Array(length);

        for (idx = 0; idx < length; idx++) {
            result[idx] = callback(array[idx], idx, array);
        }

        return result;
    };

    var operators = (function(){

        function quote(value) {
            return value.replace(quoteRegExp, "\\");
        }

        function operator(op, a, b, ignore) {
            var date;

            if (b != undefined) {
                if (typeof b === STRING) {
                    b = quote(b);
                    date = dateRegExp.exec(b);
                    if (date) {
                        b = new Date(+date[1]);
                    } else if (ignore) {
                        b = "'" + b.toLowerCase() + "'";
                        a = a + ".toLowerCase()";
                    } else {
                        b = "'" + b + "'";
                    }
                }

                if (b.getTime) {
                    //b looks like a Date
                    a = "(" + a + "?" + a + ".getTime():" + a + ")";
                    b = b.getTime();
                }
            }

            return a + " " + op + " " + b;
        }

        return {
            eq: function(a, b, ignore) {
                return operator("==", a, b, ignore);
            },
            neq: function(a, b, ignore) {
                return operator("!=", a, b, ignore);
            },
            gt: function(a, b, ignore) {
                return operator(">", a, b, ignore);
            },
            gte: function(a, b, ignore) {
                return operator(">=", a, b, ignore);
            },
            lt: function(a, b, ignore) {
                return operator("<", a, b, ignore);
            },
            lte: function(a, b, ignore) {
                return operator("<=", a, b, ignore);
            },
            startswith: function(a, b, ignore) {
                if (ignore) {
                    a = a + ".toLowerCase()";
                    if (b) {
                        b = b.toLowerCase();
                    }
                }

                if (b) {
                    b = quote(b);
                }

                return a + ".lastIndexOf('" + b + "', 0) == 0";
            },
            endswith: function(a, b, ignore) {
                if (ignore) {
                    a = a + ".toLowerCase()";
                    if (b) {
                        b = b.toLowerCase();
                    }
                }

                if (b) {
                    b = quote(b);
                }

                return a + ".lastIndexOf('" + b + "') == " + a + ".length - " + (b || "").length;
            },
            contains: function(a, b, ignore) {
                if (ignore) {
                    a = a + ".toLowerCase()";
                    if (b) {
                        b = b.toLowerCase();
                    }
                }

                if (b) {
                    b = quote(b);
                }

                return a + ".indexOf('" + b + "') >= 0"
            }
        };
    })();

    function Query(data) {
        this.data = data || [];
    }

    Query.normalizeFilter = normalizeFilter;

    Query.filterExpr = function(expression) {
        var expressions = [],
        logic = { and: " && ", or: " || " },
        idx,
        length,
        filter,
        expr,
        fieldFunctions = [],
        operatorFunctions = [],
        field,
        operator,
        filters = expression.filters;

        for (idx = 0, length = filters.length; idx < length; idx++) {
            filter = filters[idx];
            field = filter.field;
            operator = filter.operator;

            if (filter.filters) {
                expr = Query.filterExpr(filter);
                //Nested function fields or operators - update their index e.g. __o[0] -> __o[1]
                filter = expr.expression
                .replace(/__o\[(\d+)\]/g, function(match, index) {
                    index = +index;
                    return "__o[" + (operatorFunctions.length + index) + "]";
                })
                .replace(/__f\[(\d+)\]/g, function(match, index) {
                    index = +index;
                    return "__f[" + (fieldFunctions.length + index) + "]";
                });

                operatorFunctions.push.apply(operatorFunctions, expr.operators);
                fieldFunctions.push.apply(fieldFunctions, expr.fields);
            } else {
                if (typeof field === FUNCTION) {
                    expr = "__f[" + fieldFunctions.length +"](d)";
                    fieldFunctions.push(field);
                } else {
                    expr = kendo.expr(field);
                }

                if (typeof operator === FUNCTION) {
                    filter = "__o[" + operatorFunctions.length + "](" + expr + ", " + filter.value + ")";
                    operatorFunctions.push(operator);
                } else {
                    filter = operators[(operator || "eq").toLowerCase()](expr, filter.value, filter.ignoreCase !== undefined? filter.ignoreCase : true);
                }
            }

            expressions.push(filter);
        }

        return  { expression: "(" + expressions.join(logic[expression.logic]) + ")", fields: fieldFunctions, operators: operatorFunctions };
    };

    function normalizeSort(field, dir) {
        if (field) {
            var descriptor = typeof field === STRING ? { field: field, dir: dir } : field,
            descriptors = isArray(descriptor) ? descriptor : (descriptor !== undefined ? [descriptor] : []);

            return grep(descriptors, function(d) { return !!d.dir; });
        }
    }

    var operatorMap = {
        "==": "eq",
        equals: "eq",
        isequalto: "eq",
        equalto: "eq",
        equal: "eq",
        "!=": "neq",
        ne: "neq",
        notequals: "neq",
        isnotequalto: "neq",
        notequalto: "neq",
        notequal: "neq",
        "<": "lt",
        islessthan: "lt",
        lessthan: "lt",
        less: "lt",
        "<=": "lte",
        le: "lte",
        islessthanorequalto: "lte",
        lessthanequal: "lte",
        ">": "gt",
        isgreaterthan: "gt",
        greaterthan: "gt",
        greater: "gt",
        ">=": "gte",
        isgreaterthanorequalto: "gte",
        greaterthanequal: "gte",
        ge: "gte"
    };

    function normalizeOperator(expression) {
        var idx,
        length,
        filter,
        operator,
        filters = expression.filters;

        if (filters) {
            for (idx = 0, length = filters.length; idx < length; idx++) {
                filter = filters[idx];
                operator = filter.operator;

                if (operator && typeof operator === STRING) {
                    filter.operator = operatorMap[operator.toLowerCase()] || operator;
                }

                normalizeOperator(filter);
            }
        }
    }

    function normalizeFilter(expression) {
        if (expression && !isEmptyObject(expression)) {
            if (isArray(expression) || !expression.filters) {
                expression = {
                    logic: "and",
                    filters: isArray(expression) ? expression : [expression]
                }
            }

            normalizeOperator(expression);

            return expression;
        }
    }

    function normalizeAggregate(expressions) {
        return expressions = isArray(expressions) ? expressions : [expressions];
    }

    function normalizeGroup(field, dir) {
        var descriptor = typeof field === STRING ? { field: field, dir: dir } : field,
        descriptors = isArray(descriptor) ? descriptor : (descriptor !== undefined ? [descriptor] : []);

        return map(descriptors, function(d) { return { field: d.field, dir: d.dir || "asc", aggregates: d.aggregates }; });
    }

    Query.prototype = {
        toArray: function () {
            return this.data;
        },
        range: function(index, count) {
            return new Query(this.data.slice(index, index + count));
        },
        skip: function (count) {
            return new Query(this.data.slice(count));
        },
        take: function (count) {
            return new Query(this.data.slice(0, count));
        },
        select: function (selector) {
            return new Query(map(this.data, selector));
        },
        orderBy: function (selector) {
            var result = this.data.slice(0),
            comparer = isFunction(selector) || !selector ? Comparer.asc(selector) : selector.compare;

            return new Query(result.sort(comparer));
        },
        orderByDescending: function (selector) {
            return new Query(this.data.slice(0).sort(Comparer.desc(selector)));
        },
        sort: function(field, dir, comparer) {
            var idx,
            length,
            descriptors = normalizeSort(field, dir),
            comparers = [];

            comparer = comparer || Comparer;

            if (descriptors.length) {
                for (idx = 0, length = descriptors.length; idx < length; idx++) {
                    comparers.push(comparer.create(descriptors[idx]));
                }

                return this.orderBy({ compare: comparer.combine(comparers) });
            }

            return this;
        },

        filter: function(expressions) {
            var idx,
            current,
            length,
            compiled,
            predicate,
            data = this.data,
            fields,
            operators,
            result = [],
            filter;

            expressions = normalizeFilter(expressions);

            if (!expressions || expressions.filters.length === 0) {
                return this;
            }

            compiled = Query.filterExpr(expressions);
            fields = compiled.fields;
            operators = compiled.operators;

            predicate = filter = new Function("d, __f, __o", "return " + compiled.expression);

            if (fields.length || operators.length) {
                filter = function(d) {
                    return predicate(d, fields, operators);
                };
            }

            for (idx = 0, length = data.length; idx < length; idx++) {
                current = data[idx];

                if (filter(current)) {
                    result.push(current);
                }
            }
            return new Query(result);
        },

        group: function(descriptors, allData) {
            descriptors =  normalizeGroup(descriptors || []);
            allData = allData || this.data;

            var that = this,
            result = new Query(that.data),
            descriptor;

            if (descriptors.length > 0) {
                descriptor = descriptors[0];
                result = result.groupBy(descriptor).select(function(group) {
                    var data = new Query(allData).filter([ { field: group.field, operator: "eq", value: group.value } ]);
                    return {
                        field: group.field,
                        value: group.value,
                        items: descriptors.length > 1 ? new Query(group.items).group(descriptors.slice(1), data.toArray()).toArray() : group.items,
                        hasSubgroups: descriptors.length > 1,
                        aggregates: data.aggregate(descriptor.aggregates)
                    }
                });
            }
            return result;
        },

        groupBy: function(descriptor) {
            if (isEmptyObject(descriptor) || !this.data.length) {
                return new Query([]);
            }

            var field = descriptor.field,
                sorted = this._sortForGrouping(field, descriptor.dir || "asc"),
                accessor = kendo.accessor(field),
                item,
                groupValue = accessor.get(sorted[0], field),
                group = {
                    field: field,
                    value: groupValue,
                    items: []
                },
                currentValue,
                idx,
                len,
                result = [group];

            for(idx = 0, len = sorted.length; idx < len; idx++) {
                item = sorted[idx];
                currentValue = accessor.get(item, field);
                if(!groupValueComparer(groupValue, currentValue)) {
                    groupValue = currentValue;
                    group = {
                        field: field,
                        value: groupValue,
                        items: []
                    };
                    result.push(group);
                }
                group.items.push(item);
            }
            return new Query(result);
        },

        _sortForGrouping: function(field, dir) {
            var idx, length,
                data = this.data;

            if (!stableSort) {
                for (idx = 0, length = data.length; idx < length; idx++) {
                    data[idx].__position = idx;
                }

                data = new Query(data).sort(field, dir, PositionComparer).toArray();

                for (idx = 0, length = data.length; idx < length; idx++) {
                    delete data[idx].__position;
                }
                return data;
            }
            return this.sort(field, dir).toArray();
        },

        aggregate: function (aggregates) {
            var idx,
            len,
            result = {};

            if (aggregates && aggregates.length) {
                for(idx = 0, len = this.data.length; idx < len; idx++) {
                    calculateAggregate(result, aggregates, this.data[idx], idx, len);
                }
            }
            return result;
        }
    }

    function groupValueComparer(a, b) {
        if (a && a.getTime && b && b.getTime) {
            return a.getTime() === b.getTime();
        }
        return a === b;
    }

    function calculateAggregate(accumulator, aggregates, item, index, length) {
        aggregates = aggregates || [];
        var idx,
        aggr,
        functionName,
        fieldAccumulator,
        len = aggregates.length;

        for (idx = 0; idx < len; idx++) {
            aggr = aggregates[idx];
            functionName = aggr.aggregate;
            var field = aggr.field;
            accumulator[field] = accumulator[field] || {};
            accumulator[field][functionName] = functions[functionName.toLowerCase()](accumulator[field][functionName], item, kendo.accessor(field), index, length);
        }
    }

    var functions = {
        sum: function(accumulator, item, accessor) {
            return accumulator = (accumulator || 0) + accessor.get(item);
        },
        count: function(accumulator, item, accessor) {
            return (accumulator || 0) + 1;
        },
        average: function(accumulator, item, accessor, index, length) {
            accumulator = (accumulator || 0) + accessor.get(item);
            if(index == length - 1) {
                accumulator = accumulator / length;
            }
            return accumulator;
        },
        max: function(accumulator, item, accessor) {
            var accumulator =  (accumulator || 0),
            value = accessor.get(item);
            if(accumulator < value) {
                accumulator = value;
            }
            return accumulator;
        },
        min: function(accumulator, item, accessor) {
            var value = accessor.get(item),
            accumulator = (accumulator || value)
            if(accumulator > value) {
                accumulator = value;
            }
            return accumulator;
        }
    };

    function toJSON(array) {
        var idx, length = array.length, result = new Array(length);

        for (idx = 0; idx < length; idx++) {
            result[idx] = array[idx].toJSON();
        }

        return result;
    }

    function process(data, options) {
        var query = new Query(data),
        options = options || {},
        group = options.group,
        sort = normalizeGroup(group || []).concat(normalizeSort(options.sort || [])),
        total,
        filter = options.filter,
        skip = options.skip,
        take = options.take;

        if (filter) {
            query = query.filter(filter);
            total = query.toArray().length;
        }

        if (sort) {
            query = query.sort(sort);

            if (group) {
                data = query.toArray();
            }
        }

        if (skip !== undefined && take !== undefined) {
            query = query.range(skip, take);
        }

        if (group) {
            query = query.group(group, data);
        }

        return {
            total: total,
            data: query.toArray()
        };
    }

    function calculateAggregates(data, options) {
        var query = new Query(data),
        options = options || {},
        aggregates = options.aggregate,
        filter = options.filter;

        if(filter) {
            query = query.filter(filter);
        }
        return query.aggregate(aggregates);
    }

    var LocalTransport = Class.extend({
        init: function(options) {
            this.data = options.data;
        },

        read: function(options) {
            options.success(this.data);
        },
        update: function(options) {
            options.success(options.data);
        },
        create: function(options) {
            options.success(options.data);
        },
        destroy: noop
    });

    var RemoteTransport = Class.extend( {
        init: function(options) {
            var that = this, parameterMap;

            options = that.options = extend({}, that.options, options);

            each(crud, function(index, type) {
                if (typeof options[type] === STRING) {
                    options[type] = {
                        url: options[type]
                    };
                }
            });

            that.cache = options.cache? Cache.create(options.cache) : {
                find: noop,
                add: noop
            }

            parameterMap = options.parameterMap;

            that.parameterMap = isFunction(parameterMap) ? parameterMap : function(options) {
                var result = {};

                each(options, function(option, value) {
                    if (option in parameterMap) {
                        option = parameterMap[option];
                        if (isPlainObject(option)) {
                            value = option.value(value);
                            option = option.key;
                        }
                    }

                    result[option] = value;
                });

                return result;
            };
        },

        options: {
            parameterMap: identity
        },

        create: function(options) {
            return ajax(this.setup(options, CREATE));
        },

        read: function(options) {
            var that = this,
                success,
                error,
                result,
                cache = that.cache;

            options = that.setup(options, READ);

            success = options.success || noop;
            error = options.error || noop;

            result = cache.find(options.data);

            if(result !== undefined) {
                success(result);
            } else {
                options.success = function(result) {
                    cache.add(options.data, result);

                    success(result);
                };

                $.ajax(options);
            }
        },

        update: function(options) {
            return ajax(this.setup(options, UPDATE));
        },

        destroy: function(options) {
            return ajax(this.setup(options, DESTROY));
        },

        setup: function(options, type) {
            options = options || {};

            var that = this,
                parameters,
                operation = that.options[type],
                data = isFunction(operation.data) ? operation.data() : operation.data;

            options = extend(true, {}, operation, options);
            parameters = extend(data, options.data);

            options.data = that.parameterMap(parameters, type);

            if (isFunction(options.url)) {
                options.url = options.url(parameters);
            }

            return options;
        }
    });

    Cache.create = function(options) {
        var store = {
            "inmemory": function() { return new Cache(); }
        };

        if (isPlainObject(options) && isFunction(options.find)) {
            return options;
        }

        if (options === true) {
            return new Cache();
        }

        return store[options]();
    }

    function Cache() {
        this._store = {};
    }

    Cache.prototype = /** @ignore */ {
        add: function(key, data) {
            if(key !== undefined) {
                this._store[stringify(key)] = data;
            }
        },
        find: function(key) {
            return this._store[stringify(key)];
        },
        clear: function() {
            this._store = {};
        },
        remove: function(key) {
            delete this._store[stringify(key)];
        }
    }

    var DataReader = Class.extend({
        init: function(schema) {
            var that = this, member, get, model;

            schema = schema || {};

            for (member in schema) {
                get = schema[member];

                that[member] = typeof get === STRING ? getter(get) : get;
            }

            if (isPlainObject(that.model)) {
                that.model = model = kendo.data.Model.define(that.model);

                var dataFunction = that.data,
                getters = {};

                if (model.fields) {
                    each(model.fields, function(field, value) {
                        if (isPlainObject(value) && value.field) {
                            getters[value.field] = getter(value.field);
                        } else {
                            getters[field] = getter(field);
                        }
                    });
                }

                that.data = function(data) {
                    var record,
                        getter,
                        idx,
                        length,
                        modelInstance = new that.model();

                    data = dataFunction(data);

                    if (!isEmptyObject(getters)) {
                        for (idx = 0, length = data.length; idx < length; idx++) {
                            record = data[idx];
                            for (getter in getters) {
                                record[getter] = modelInstance._parse(getter, getters[getter](record));
                            }
                        }
                    }

                    return data;
                }
            }
        },
        parse: identity,
        data: identity,
        total: function(data) {
            return data.length;
        },
        groups: identity,
        status: function(data) {
            return data.status;
        },
        aggregates: function() {
            return {};
        }
    });

    var DataSource = Observable.extend({
        init: function(options) {
            var that = this, id, model, transport;

            options = that.options = extend({}, that.options, options);

            extend(that, {
                _map: {},
                _prefetch: {},
                _data: [],
                _ranges: [],
                _view: [],
                _pristine: [],
                _destroyed: [],
                _pageSize: options.pageSize,
                _page: options.page  || (options.pageSize ? 1 : undefined),
                _sort: normalizeSort(options.sort),
                _filter: normalizeFilter(options.filter),
                _group: normalizeGroup(options.group),
                _aggregate: options.aggregate
            });

            Observable.fn.init.call(that);

            transport = options.transport;

            if (transport) {
                transport.read = typeof transport.read === STRING ? { url: transport.read } : transport.read;

                if (options.type) {
                    transport = extend(true, {}, kendo.data.transports[options.type], transport);
                    options.schema = extend(true, {}, kendo.data.schemas[options.type], options.schema);
                }

                that.transport = isFunction(transport.read) ? transport: new RemoteTransport(transport);
            } else {
                that.transport = new LocalTransport({ data: options.data });
            }

            that.reader = new kendo.data.readers[options.schema.type || "json" ](options.schema);

            model = that.reader.model || {};

            that._data = that._observe(that._data);

            id = model.id;

            if (id) {
                that.id = function(record) {
                    return id(record);
                };
            }

            that.bind([ERROR, CHANGE, REQUESTSTART], options);
        },

        options: {
            data: [],
            schema: {},
            serverSorting: false,
            serverPaging: false,
            serverFiltering: false,
            serverGrouping: false,
            serverAggregates: false,
            sendAllFields: true,
            batch: false
        },

        get: function(id) {
            var idx, length, data = this._data;

            for (idx = 0, length = data.length; idx < length; idx++) {
                if (data[idx].id == id) {
                    return data[idx];
                }
            }
        },

        getByUid: function(id) {
            var idx, length, data = this._data;

            for (idx = 0, length = data.length; idx < length; idx++) {
                if (data[idx].uid == id) {
                    return data[idx];
                }
            }
        },

        sync: function() {
            var that = this,
            idx,
            length,
            created = [],
            updated = [],
            destroyed = that._destroyed,
            data = that._data;

            if (!that.reader.model) {
                return;
            }

            for (idx = 0, length = data.length; idx < length; idx++) {
                if (data[idx].isNew()) {
                    created.push(data[idx]);
                } else if (data[idx].dirty) {
                    updated.push(data[idx]);
                }
            }

            var promises = that._send("create", created);

            promises.push.apply(promises ,that._send("update", updated));
            promises.push.apply(promises ,that._send("destroy", destroyed));

            $.when.apply(null, promises)
            .then(function() {
                var idx,
                length;

                for (idx = 0, length = arguments.length; idx < length; idx++){
                    that._accept(arguments[idx]);
                }

                that._change();
            });

        },

        _accept: function(result) {
            var that = this,
            models = result.models,
            response = result.response,
            idx = 0,
            pristine = that.reader.data(that._pristine),
            type = result.type,
            length;

            if (response) {
                response = that.reader.data(that.reader.parse(response));
            } else {
                response = $.map(models, function(model) { return model.toJSON(); } );
            }

            if (!$.isArray(response)) {
                response = [response];
            }

            if (type === "destroy") {
                that._destroyed = [];
            }

            for (idx = 0, length = models.length; idx < length; idx++) {
                if (type !== "destroy") {
                    models[idx].accept(response[idx]);

                    if (type === "create") {
                        pristine.push(models[idx]);
                    } else if (type === "update") {
                        extend(pristine[that._pristineIndex(models[idx])], response[idx]);
                    }
                } else {
                    pristine.splice(that._pristineIndex(models[idx]), 1);
                }
            }
        },

        _pristineIndex: function(model) {
            var that = this,
            idx,
            length,
            pristine = that.reader.data(that._pristine);

            for (idx = 0, length = pristine.length; idx < length; idx++) {
                if (pristine[idx][model.idField] === model.id) {
                    return idx;
                }
            }
            return -1;
        },

        _promise: function(data, models, type) {
            var that = this,
            transport = that.transport;

            return $.Deferred(function(deferred) {
                transport[type].call(transport, extend({
                    success: function(response) {
                        deferred.resolve({
                            response: response,
                            models: models,
                            type: type
                        });
                    },
                    error: function(response) {
                        deferred.reject(response);
                        that.trigger(ERROR, response);
                    }
                }, data)
                );
            }).promise();
        },

        _send: function(method, data) {
            var that = this,
            idx,
            length,
            promises = [],
            transport = that.transport;

            if (that.options.batch) {
                if (data.length) {
                    promises.push(that._promise( { data: { models: toJSON(data) } }, data , method));
                }
            } else {
                for (idx = 0, length = data.length; idx < length; idx++) {
                    promises.push(that._promise( { data: data[idx].toJSON() }, [ data[idx] ], method));
                }
            }

            return promises;
        },

        add: function(model) {
            return this.insert(this._data.length, model);
        },

        insert: function(index, model) {
            if (!model) {
                model = index;
                index = 0;
            }

            if (!(model instanceof Model)) {
                if (this.reader.model) {
                    model = new this.reader.model(model);
                } else {
                    model = new ObservableObject(model);
                }
            }

            this._data.splice(index, 0, model);

            return model;
        },

        cancelChanges: function(model) {
            var that = this,
                pristineIndex,
                pristine = that.reader.data(that._pristine),
                index;

            if (model instanceof kendo.data.Model) {
                index = that.indexOf(model);
                pristineIndex = that._pristineIndex(model);
                if (index != -1) {
                    if (pristineIndex != -1 && !model.isNew()) {
                        extend(true, that._data[index], pristine[pristineIndex]);
                    } else {
                        that._data.splice(index, 1);
                    }
                }
            } else {
                that._data = that._observe(pristine);
                that._change();
            }
        },

        read: function(data) {
            var that = this, params = that._params(data);

            that._queueRequest(params, function() {
                that.trigger(REQUESTSTART);
                that._ranges = [];
                that.transport.read({
                    data: params,
                    success: proxy(that.success, that),
                    error: proxy(that.error, that)
                });
            });
        },

        indexOf: function(model) {
            var idx, length, data = this._data;

            if (model) {
                for (idx = 0, length = data.length; idx < length; idx++) {
                    if (data[idx].uid == model.uid) {
                        return idx;
                    }
                }
            }
            return -1;
        },

        _params: function(data) {
            var that = this,
            options =  extend({
                take: that.take(),
                skip: that.skip(),
                page: that.page(),
                pageSize: that.pageSize(),
                sort: that._sort,
                filter: that._filter,
                group: that._group,
                aggregate: that._aggregate
            }, data);

            if (!that.options.serverPaging) {
                delete options.take;
                delete options.skip;
                delete options.page;
                delete options.pageSize;
            }
            return options;
        },

        _queueRequest: function(options, callback) {
            var that = this;
            if (!that._requestInProgress) {
                that._requestInProgress = true;
                that._pending = undefined;
                callback();
            } else {
                that._pending = { callback: proxy(callback, that), options: options };
            }
        },

        _dequeueRequest: function() {
            var that = this;
            that._requestInProgress = false;
            if (that._pending) {
                that._queueRequest(that._pending.options, that._pending.callback);
            }
        },

        remove: function(model) {
            var idx, length, data = this._data;

            for (idx = 0, length = data.length; idx < length; idx++) {
                if (data[idx].uid == model.uid) {
                    model = data[idx];
                    data.splice(idx, 1);
                    return model;
                }
            }
        },

        error: function(xhr, status, errorThrown) {
            this.trigger(ERROR, { xhr: xhr, status: status, errorThrown: errorThrown });
        },

        success: function(data) {
            var that = this,
            options = {},
            result,
            hasGroups = that.options.serverGrouping === true && that._group && that._group.length > 0;

            data = that.reader.parse(data);

            that._pristine = isPlainObject(data) ? $.extend(true, {}, data) : data.slice(0);

            that._total = that.reader.total(data);

            if (that._aggregate && that.options.serverAggregates) {
                that._aggregateResult = that.reader.aggregates(data);
            }

            if (hasGroups) {
                data = that.reader.groups(data);
            } else {
                data = that.reader.data(data);
            }

            that._data = that._observe(data);

            var start = that._skip || 0,
            end = start + that._data.length;

            that._ranges.push({ start: start, end: end, data: that._data });
            that._ranges.sort( function(x, y) { return x.start - y.start; } );

            that._dequeueRequest();
            that._process(that._data);
        },

        _observe: function(data) {
            var that = this,
                model = that.reader.model,
                wrap = false;

            if (model && data.length) {
                wrap = !(data[0] instanceof model);
            }

            if (data instanceof ObservableArray) {
                if (wrap) {
                    data.type = that.reader.model;
                    data.wrapAll(data, data);
                }
            } else {
                data = new ObservableArray(data, that.reader.model);
            }

            return data.bind(CHANGE, proxy(that._change, that));
        },

        _change: function(e) {
            var that = this, idx, length, action = e ? e.action : "";

            if (action === "remove") {
                for (idx = 0, length = e.items.length; idx < length; idx++) {
                    if (!e.items[idx].isNew || !e.items[idx].isNew()) {
                        that._destroyed.push(e.items[idx]);
                    }
                }
            }

            if (that.options.autoSync && (action === "add" || action === "remove" || action === "itemchange")) {
                that.sync();
            } else {
                var total = that._total || that.reader.total(that._pristine);
                if (action === "add") {
                    total++;
                } else if (action === "remove") {
                    total--;
                } else if (action !== "itemchange" && !that.options.serverPaging) {
                    total = that.reader.total(that._pristine);
                }

                that._total = total;
                that._process(that._data, e);
            }
        },

        _process: function (data, e) {
            var that = this,
            options = {},
            result,
            hasGroups = that.options.serverGrouping === true && that._group && that._group.length > 0;

            if (that.options.serverPaging !== true) {
                options.skip = that._skip;
                options.take = that._take || that._pageSize;

                if(options.skip === undefined && that._page !== undefined && that._pageSize !== undefined) {
                    options.skip = (that._page - 1) * that._pageSize;
                }
            }

            if (that.options.serverSorting !== true) {
                options.sort = that._sort;
            }

            if (that.options.serverFiltering !== true) {
                options.filter = that._filter;
            }

            if (that.options.serverGrouping !== true) {
                options.group = that._group;
            }

            if (that.options.serverAggregates !== true) {
                options.aggregate = that._aggregate;
                that._aggregateResult = calculateAggregates(data, options);
            }

            result = process(data, options);

            that._view = result.data;

            if (result.total !== undefined && !that.options.serverFiltering) {
                that._total = result.total;
            }

            that.trigger(CHANGE, e);
        },

        at: function(index) {
            return this._data[index];
        },

        data: function(value) {
            var that = this;
            if (value !== undefined) {
                that._data = this._observe(value);

                that._total = that._data.length;

                that._process(that._data);
            } else {
                return that._data;
            }
        },

        view: function() {
            return this._view;
        },

        query: function(options) {
            var that = this,
            result,
            remote = that.options.serverSorting || that.options.serverPaging || that.options.serverFiltering || that.options.serverGrouping || that.options.serverAggregates;

            if (options !== undefined) {
                that._pageSize = options.pageSize;
                that._page = options.page;
                that._sort = options.sort;
                that._filter = options.filter;
                that._group = options.group;
                that._aggregate = options.aggregate;
                that._skip = options.skip;
                that._take = options.take;

                if(that._skip === undefined) {
                    that._skip = that.skip();
                    options.skip = that.skip();
                }

                if(that._take === undefined && that._pageSize !== undefined) {
                    that._take = that._pageSize;
                    options.take = that._take;
                }

                if (options.sort) {
                    that._sort = options.sort = normalizeSort(options.sort);
                }

                if (options.filter) {
                    that._filter = options.filter = normalizeFilter(options.filter);
                }

                if (options.group) {
                    that._group = options.group = normalizeGroup(options.group);
                }
                if (options.aggregate) {
                    that._aggregate = options.aggregate = normalizeAggregate(options.aggregate);
                }
            }

            if (remote || (that._data === undefined || that._data.length == 0)) {
                that.read(options);
            } else {
                that.trigger(REQUESTSTART);
                result = process(that._data, options);

                if (!that.options.serverFiltering) {
                    if (result.total !== undefined) {
                        that._total = result.total;
                    } else {
                        that._total = that._data.length;
                    }
                }

                that._view = result.data;
                that._aggregateResult = calculateAggregates(that._data, options);
                that.trigger(CHANGE);
            }
        },

        fetch: function(callback) {
            var that = this;

            if (callback && isFunction(callback)) {
                that.one(CHANGE, callback);
            }

            that._query();
        },

        _query: function(options) {
            var that = this;

            that.query(extend({}, {
                page: that.page(),
                pageSize: that.pageSize(),
                sort: that.sort(),
                filter: that.filter(),
                group: that.group(),
                aggregate: that.aggregate()
            }, options));
        },

        page: function(val) {
            var that = this,
            skip;

            if(val !== undefined) {
                val = math.max(math.min(math.max(val, 1), that.totalPages()), 1);
                that._query({ page: val });
                return;
            }
            skip = that.skip();

            return skip !== undefined ? math.round((skip || 0) / (that.take() || 1)) + 1 : undefined;
        },

        pageSize: function(val) {
            var that = this;

            if(val !== undefined) {
                that._query({ pageSize: val });
                return;
            }

            return that.take();
        },

        sort: function(val) {
            var that = this;

            if(val !== undefined) {
                that._query({ sort: val });
                return;
            }

            return that._sort;
        },

        filter: function(val) {
            var that = this;

            if (val === undefined) {
                return that._filter;
            }

            that._query({ filter: val, page: 1 });
        },

        group: function(val) {
            var that = this;

            if(val !== undefined) {
                that._query({ group: val });
                return;
            }

            return that._group;
        },

        total: function() {
            return this._total || 0;
        },

        aggregate: function(val) {
            var that = this;

            if(val !== undefined) {
                that._query({ aggregate: val });
                return;
            }

            return that._aggregate;
        },

        aggregates: function() {
            return this._aggregateResult;
        },

        totalPages: function() {
            var that = this,
            pageSize = that.pageSize() || that.total();

            return math.ceil((that.total() || 0) / pageSize);
        },

        inRange: function(skip, take) {
            var that = this,
            end = math.min(skip + take, that.total());

            if (!that.options.serverPaging && that.data.length > 0) {
                return true;
            }

            return that._findRange(skip, end).length > 0;
        },

        range: function(skip, take) {
            skip = math.min(skip || 0, this.total());
            var that = this,
            pageSkip = math.max(math.floor(skip / take), 0) * take,
            size = math.min(pageSkip + take, that.total()),
            data;

            data = that._findRange(skip, math.min(skip + take, that.total()));

            if (data.length) {
                that._skip = skip > that.skip() ? math.min(size, (that.totalPages() - 1) * that.take()) : pageSkip;

                that._take = take;

                var paging = that.options.serverPaging;
                var sorting = that.options.serverSorting;
                try {
                    that.options.serverPaging = true;
                    that.options.serverSorting = true;
                    that._process(data);
                } finally {
                    that.options.serverPaging = paging;
                    that.options.serverSorting = sorting;
                }

                return;
            }

            if (take !== undefined) {
                if (!that._rangeExists(pageSkip, size)) {
                    that.prefetch(pageSkip, take, function() {
                        if (skip > pageSkip && size < that.total() && !that._rangeExists(size, math.min(size + take, that.total()))) {
                            that.prefetch(size, take, function() {
                                that.range(skip, take);
                            });
                        } else {
                            that.range(skip, take);
                        }
                    });
                } else if (pageSkip < skip) {
                    that.prefetch(size, take, function() {
                        that.range(skip, take);
                    });
                }
            }
        },

        _findRange: function(start, end) {
            var that = this,
                ranges = that._ranges,
                range,
                data = [],
                skipIdx,
                takeIdx,
                startIndex,
                endIndex,
                rangeData,
                rangeEnd,
                processed,
                options = that.options,
                remote = options.serverSorting || options.serverPaging || options.serverFiltering || options.serverGrouping || options.serverAggregates,
                length;

            for (skipIdx = 0, length = ranges.length; skipIdx < length; skipIdx++) {
                range = ranges[skipIdx];
                if (start >= range.start && start <= range.end) {
                    var count = 0;

                    for (takeIdx = skipIdx; takeIdx < length; takeIdx++) {
                        range = ranges[takeIdx];

                        if (range.data.length && start + count >= range.start /*&& count + count <= range.end*/) {
                            rangeData = range.data;
                            rangeEnd = range.end;

                            if (!remote) {
                                processed = process(range.data, { sort: that.sort(), filter: that.filter() });
                                rangeData = processed.data;

                                if (processed.total !== undefined) {
                                    rangeEnd = processed.total;
                                }
                            }

                            startIndex = 0;
                            if (start + count > range.start) {
                                startIndex = (start + count) - range.start;
                            }
                            endIndex = rangeData.length;
                            if (rangeEnd > end) {
                                endIndex = endIndex - (rangeEnd - end);
                            }
                            count += endIndex - startIndex;
                            data = data.concat(rangeData.slice(startIndex, endIndex));

                            if (end <= range.end && count == end - start) {
                                return data;
                            }
                        }
                    }
                    break;
                }
            }
            return [];
        },

        skip: function() {
            var that = this;

            if (that._skip === undefined) {
                return (that._page !== undefined ? (that._page  - 1) * (that.take() || 1) : undefined);
            }
            return that._skip;
        },

        take: function() {
            var that = this;
            return that._take || that._pageSize;
        },

        prefetch: function(skip, take, callback) {
            var that = this,
            size = math.min(skip + take, that.total()),
            range = { start: skip, end: size, data: [] },
            options = {
                take: take,
                skip: skip,
                page: skip / take + 1,
                pageSize: take,
                sort: that._sort,
                filter: that._filter,
                group: that._group,
                aggregate: that._aggregate
            };

            if (!that._rangeExists(skip, size)) {
                clearTimeout(that._timeout);

                that._timeout = setTimeout(function() {
                    that._queueRequest(options, function() {
                        that.transport.read({
                            data: options,
                            success: function (data) {
                                that._dequeueRequest();
                                var found = false;
                                for (var i = 0, len = that._ranges.length; i < len; i++) {
                                    if (that._ranges[i].start === skip) {
                                        found = true;
                                        range = that._ranges[i];
                                        break;
                                    }
                                }
                                if (!found) {
                                    that._ranges.push(range);
                                }

                                data = that.reader.parse(data);
                                range.data = that._observe(that.reader.data(data));
                                range.end = range.start + range.data.length;
                                that._ranges.sort( function(x, y) { return x.start - y.start; } );
                                that._total = that.reader.total(data);
                                if (callback) {
                                    callback();
                                }
                            }
                        });
                    });
                }, 100);
            } else if (callback) {
                callback();
            }
        },

        _rangeExists: function(start, end) {
            var that = this,
            ranges = that._ranges,
            idx,
            length;

            for (idx = 0, length = ranges.length; idx < length; idx++) {
                if (ranges[idx].start <= start && ranges[idx].end >= end) {
                    return true;
                }
            }
            return false;
        }
    });

    DataSource.create = function(options) {
        options = options && options.push ? { data: options } : options;

        var dataSource = options || {},
        data = dataSource.data,
        fields = dataSource.fields,
        table = dataSource.table,
        select = dataSource.select,
        idx,
        length,
        model = {},
        field;

        if (!data && fields && !dataSource.transport) {
            if (table) {
                data = inferTable(table, fields);
            } else if (select) {
                data = inferSelect(select, fields);
            }
        }

        if (kendo.data.Model && fields && (!dataSource.schema || !dataSource.schema.model)) {
            for (idx = 0, length = fields.length; idx < length; idx++) {
                field = fields[idx];
                if (field.type) {
                    model[field.field] = field;
                }
            }

            if (!isEmptyObject(model)) {
                dataSource.schema = extend(true, dataSource.schema, { model:  { fields: model } });
            }
        }

        dataSource.data = data;

        return dataSource instanceof DataSource ? dataSource : new DataSource(dataSource);
    }

    function inferSelect(select, fields) {
        var options = $(select)[0].children,
            idx,
            length,
            data = [],
            record,
            firstField = fields[0],
            secondField = fields[1],
            value,
            option;

        for (idx = 0, length = options.length; idx < length; idx++) {
            record = {};
            option = options[idx];

            record[firstField.field] = option.text;

            value = option.attributes.value;

            if (value && value.specified) {
                value = option.value;
            } else {
                value = option.text;
            }

            record[secondField.field] = value;

            data.push(record);
        }

        return data;
    }

    function inferTable(table, fields) {
        var tbody = $(table)[0].tBodies[0],
        rows = tbody ? tbody.rows : [],
        idx,
        length,
        fieldIndex,
        fieldCount = fields.length,
        data = [],
        cells,
        record,
        cell,
        empty;

        for (idx = 0, length = rows.length; idx < length; idx++) {
            record = {};
            empty = true;
            cells = rows[idx].cells;

            for (fieldIndex = 0; fieldIndex < fieldCount; fieldIndex++) {
                cell = cells[fieldIndex];
                if(cell.nodeName.toLowerCase() !== "th") {
                    empty = false;
                    record[fields[fieldIndex].field] = cell.innerHTML;
                }
            }
            if(!empty) {
                data.push(record);
            }
        }

        return data;
    }

    extend(true, kendo.data, /** @lends kendo.data */ {
        readers: {
            json: DataReader
        },
        Query: Query,
        DataSource: DataSource,
        ObservableObject: ObservableObject,
        ObservableArray: ObservableArray,
        LocalTransport: LocalTransport,
        RemoteTransport: RemoteTransport,
        Cache: Cache,
        DataReader: DataReader,
        Model: Model
    });
})(jQuery);
/*
* Kendo UI Web v2012.1.322 (http://kendoui.com)
* Copyright 2012 Telerik AD. All rights reserved.
*
* Kendo UI Web commercial licenses may be obtained at http://kendoui.com/web-license
* If you do not own a commercial license, this file shall be governed by the
* GNU General Public License (GPL) version 3.
* For GPL requirements, please review: http://www.gnu.org/copyleft/gpl.html
*/
(function($, undefined) {
    var kendo = window.kendo,
        extend = $.extend,
        odataFilters = {
            eq: "eq",
            neq: "ne",
            gt: "gt",
            gte: "ge",
            lt: "lt",
            lte: "le",
            contains : "substringof",
            endswith: "endswith",
            startswith: "startswith"
        },
        mappers = {
            pageSize: $.noop,
            page: $.noop,
            filter: function(params, filter) {
                if (filter) {
                    params.$filter = toOdataFilter(filter);
                }
            },
            sort: function(params, orderby) {
                params.$orderby = $.map(orderby, function(value) {
                    var order = value.field.replace(/\./g, "/");

                    if (value.dir === "desc") {
                        order += " desc";
                    }

                    return order;
                }).join(",");
            },
            skip: function(params, skip) {
                if (skip) {
                    params.$skip = skip;
                }
            },
            take: function(params, take) {
                if (take) {
                    params.$top = take;
                }
            }
        },
        defaultDataType = {
            read: {
                dataType: "jsonp"
            }
        };

    function toOdataFilter(filter) {
        var result = [],
            logic = filter.logic || "and",
            idx,
            length,
            field,
            type,
            format,
            operator,
            value,
            filters = filter.filters;

        for (idx = 0, length = filters.length; idx < length; idx++) {
            filter = filters[idx];
            field = filter.field;
            value = filter.value;
            operator = filter.operator;

            if (filter.filters) {
                filter = toOdataFilter(filter);
            } else {
                field = field.replace(/\./g, "/"),

                filter = odataFilters[operator];

                if (filter && value !== undefined) {
                    type = $.type(value);
                    if (type === "string") {
                        format = "'{1}'";
                        value = value.replace(/'/g, "''");
                    } else if (type === "date") {
                        format = "datetime'{1:yyyy-MM-ddTHH:mm:ss}'";
                    } else {
                        format = "{1}";
                    }

                    if (filter.length > 3) {
                        if (filter !== "substringof") {
                            format = "{0}({2}," + format + ")";
                        } else {
                            format = "{0}(" + format + ",{2})";
                        }
                    } else {
                        format = "{2} {0} " + format;
                    }

                    filter = kendo.format(format, filter, value, field);
                }
            }

            result.push(filter);
        }

        filter = result.join(" " + logic + " ");

        if (result.length > 1) {
            filter = "(" + filter + ")";
        }

        return filter;
    }

    extend(true, kendo.data, {
        schemas: {
            odata: {
                type: "json",
                data: function(data) {
                    return data.d.results || [data.d];
                },
                total: "d.__count"
            }
        },
        transports: {
            odata: {
                read: {
                    cache: true, // to prevent jQuery from adding cache buster
                    dataType: "jsonp",
                    jsonp: "$callback"
                },
                update: {
                    cache: true,
                    dataType: "json",
                    contentType: "application/json", // to inform the server the the request body is JSON encoded
                    type: "PUT" // can be PUT or MERGE
                },
                create: {
                    cache: true,
                    dataType: "json",
                    contentType: "application/json",
                    type: "POST" // must be POST to create new entity
                },
                destroy: {
                    cache: true,
                    dataType: "json",
                    type: "DELETE"
                },
                parameterMap: function(options, type) {
                    var params,
                        value,
                        option,
                        dataType;

                    options = options || {};
                    type = type || "read";
                    dataType = (this.options || defaultDataType)[type];
                    dataType = dataType ? dataType.dataType : "json";

                    if (type === "read") {
                        params = {
                            $format: "json",
                            $inlinecount: "allpages"
                        };

                        for (option in options) {
                            if (mappers[option]) {
                                mappers[option](params, options[option]);
                            } else {
                                params[option] = options[option];
                            }
                        }
                    } else {
                        if (dataType !== "json") {
                            throw new Error("Only json dataType can be used for " + type + " operation.");
                        }

                        if (type !== "destroy") {
                            for (option in options) {
                                value = options[option];
                                if (typeof value === "number") {
                                    options[option] = value + "";
                                }
                            }

                            params = kendo.stringify(options);
                        }
                    }

                    return params;
                }
            }
        }
    });
})(jQuery);
/*
* Kendo UI Web v2012.1.322 (http://kendoui.com)
* Copyright 2012 Telerik AD. All rights reserved.
*
* Kendo UI Web commercial licenses may be obtained at http://kendoui.com/web-license
* If you do not own a commercial license, this file shall be governed by the
* GNU General Public License (GPL) version 3.
* For GPL requirements, please review: http://www.gnu.org/copyleft/gpl.html
*/
(function($, undefined) {
    var kendo = window.kendo,
        ui = kendo.ui,
        Widget = ui.Widget,
        proxy = $.proxy;

    function button(template, idx, text, numeric) {
        return template( {
            idx: idx,
            text: text,
            ns: kendo.ns,
            numeric: numeric
        });
    }

    var Pager = Widget.extend( {
        init: function(element, options) {
            var that = this;

            Widget.fn.init.call(that, element, options);

            options = that.options;
            that.dataSource = options.dataSource;
            that.linkTemplate = kendo.template(that.options.linkTemplate);
            that.selectTemplate = kendo.template(that.options.selectTemplate);

            that._refreshHandler = proxy(that.refresh, that);

            that.dataSource.bind("change", that._refreshHandler);

            that.list = $('<ul class="k-pager k-reset k-numeric" />').appendTo(that.element);//.html(that.selectTemplate({ text: 1 }));
            that._clickHandler = proxy(that._click, that);

            that.element.delegate("a", "click", that._clickHandler);

            that.refresh();
        },

        destroy: function() {
            var that = this;

            that.element.undelegate("a", "click", that._clickHandler);
            that.dataSource.unbind("change", that._refreshHandler);
        },

        options: {
            name: "Pager",
            selectTemplate: '<li><span class="k-state-active">#=text#</span></li>',
            linkTemplate: '<li><a href="\\#" class="k-link" data-#=ns#page="#=idx#">#=text#</a></li>',
            buttonCount: 10
        },

        refresh: function() {
            var that = this,
                idx,
                end,
                start = 1,
                html = "",
                reminder,
                page = that.page(),
                totalPages = that.totalPages(),
                linkTemplate = that.linkTemplate,
                buttonCount = that.options.buttonCount;

            if (page > buttonCount) {
                reminder = (page % buttonCount);

                start = (reminder == 0) ? (page - buttonCount) + 1 : (page - reminder) + 1;
            }

            end = Math.min((start + buttonCount) - 1, totalPages);

            if(start > 1) {
                html += button(linkTemplate, start - 1, "...", false);
            }

            for(idx = start; idx <= end; idx++) {
                html += button(idx == page ? that.selectTemplate : linkTemplate, idx, idx, true);
            }

            if(end < totalPages) {
                html += button(linkTemplate, idx, "...", false);
            }

            if (html === "") {
                html = that.selectTemplate({ text: 1 });
            }

            that.list.empty().append(html);
        },

        _click: function(e) {
            var page = $(e.currentTarget).attr(kendo.attr("page"));
            e.preventDefault();

            this.dataSource.page(page);

            this.trigger("change", { index: page });
        },

        totalPages: function() {
            return Math.ceil((this.dataSource.total() || 0) / this.pageSize());
        },

        pageSize: function() {
            return this.dataSource.pageSize() || this.dataSource.total();
        },

        page: function() {
            return this.dataSource.page() || 1;
        }
    });

    ui.plugin(Pager);
})(jQuery);
/*
* Kendo UI Web v2012.1.322 (http://kendoui.com)
* Copyright 2012 Telerik AD. All rights reserved.
*
* Kendo UI Web commercial licenses may be obtained at http://kendoui.com/web-license
* If you do not own a commercial license, this file shall be governed by the
* GNU General Public License (GPL) version 3.
* For GPL requirements, please review: http://www.gnu.org/copyleft/gpl.html
*/
(function($, undefined) {
    var kendo = window.kendo,
        ui = kendo.ui,
        Widget = ui.Widget,
        proxy = $.proxy,
        isFunction = $.isFunction,
        extend = $.extend,
        HORIZONTAL = "horizontal",
        VERTICAL = "vertical",
        START = "start",
        RESIZE = "resize",
        RESIZEEND = "resizeend";

    var Resizable = Widget.extend({
        init: function(element, options) {
            var that = this;

            Widget.fn.init.call(that, element, options);

            that.orientation = that.options.orientation.toLowerCase() != VERTICAL ? HORIZONTAL : VERTICAL;
            that._positionMouse = that.orientation == HORIZONTAL ? "pageX" : "pageY";
            that._position = that.orientation == HORIZONTAL ? "left" : "top";
            that._sizingDom = that.orientation == HORIZONTAL ? "outerWidth" : "outerHeight";

            new ui.Draggable(element, {
                distance: 0,
                filter: options.handle,
                drag: proxy(that._resize, that),
                dragstart: proxy(that._start, that),
                dragend: proxy(that._stop, that)
            });
        },

        events: [
            RESIZE,
            RESIZEEND,
            START
        ],

        options: {
            name: "Resizable",
            orientation: HORIZONTAL
        },
        _max: function(e) {
            var that = this,
                hintSize = that.hint ? that.hint[that._sizingDom]() : 0,
                size = that.options.max;

            return isFunction(size) ? size(e) : size !== undefined ? (that._initialElementPosition + size) - hintSize : size;
        },
        _min: function(e) {
            var that = this,
                size = that.options.min;

            return isFunction(size) ? size(e) : size !== undefined ? that._initialElementPosition + size : size;
        },
        _start: function(e) {
            var that = this,
                hint = that.options.hint,
                el = $(e.currentTarget);

            that._initialMousePosition = e[that._positionMouse];
            that._initialElementPosition = el.position()[that._position];

            if (hint) {
                that.hint = isFunction(hint) ? $(hint(el)) : hint;

                that.hint.css({
                    position: "absolute"
                })
                .css(that._position, that._initialElementPosition)
                .appendTo(that.element);
            }

            that.trigger(START, e);

            that._maxPosition = that._max(e);
            that._minPosition = that._min(e);

            $(document.body).css("cursor", el.css("cursor"));
        },
        _resize: function(e) {
            var that = this,
                handle = $(e.currentTarget),
                maxPosition = that._maxPosition,
                minPosition = that._minPosition,
                currentPosition = that._initialElementPosition + (e[that._positionMouse] - that._initialMousePosition),
                position;

            position = minPosition !== undefined ? Math.max(minPosition, currentPosition) : currentPosition;
            that.position = position =  maxPosition !== undefined ? Math.min(maxPosition, position) : position;

            if(that.hint) {
                that.hint.toggleClass(that.options.invalidClass || "", position == maxPosition || position == minPosition)
                         .css(that._position, position);
            }

            that.trigger(RESIZE, extend(e, { position: position }));
        },
        _stop: function(e) {
            var that = this;

            if(that.hint) {
                that.hint.remove();
            }

            that.trigger(RESIZEEND, extend(e, { position: that.position }));
            $(document.body).css("cursor", "");
        }
    });

    kendo.ui.plugin(Resizable);

})(jQuery);
/*
* Kendo UI Web v2012.1.322 (http://kendoui.com)
* Copyright 2012 Telerik AD. All rights reserved.
*
* Kendo UI Web commercial licenses may be obtained at http://kendoui.com/web-license
* If you do not own a commercial license, this file shall be governed by the
* GNU General Public License (GPL) version 3.
* For GPL requirements, please review: http://www.gnu.org/copyleft/gpl.html
*/
(function($, undefined) {
    var kendo = window.kendo,
        proxy = $.proxy,
        DIR = "dir",
        ASC = "asc",
        SINGLE = "single",
        FIELD = "field",
        DESC = "desc",
        TLINK = ".k-link",
        Widget = kendo.ui.Widget;

    var Sortable = Widget.extend({
        init: function(element, options) {
            var that = this, link;

            Widget.fn.init.call(that, element, options);

            that.dataSource = that.options.dataSource.bind("change", proxy(that.refresh, that));
            link = that.element.find(TLINK);

            if (!link[0]) {
                link = that.element.wrapInner('<a class="k-link" href="#"/>').find(TLINK);
            }

            that.link = link;
            that.element.click(proxy(that._click, that));
        },

        options: {
            name: "Sortable",
            mode: SINGLE,
            allowUnsort: true
        },

        refresh: function() {
            var that = this,
                sort = that.dataSource.sort() || [],
                idx,
                length,
                descriptor,
                dir,
                element = that.element,
                field = element.data(FIELD);

            element.removeData(DIR);

            for (idx = 0, length = sort.length; idx < length; idx++) {
               descriptor = sort[idx];

               if (field == descriptor.field) {
                   element.data(DIR, descriptor.dir);
               }
            }

            dir = element.data(DIR);

            element.find(".k-arrow-up,.k-arrow-down").remove();

            if (dir === ASC) {
                $('<span class="k-icon k-arrow-up" />').appendTo(that.link);
            } else if (dir === DESC) {
                $('<span class="k-icon k-arrow-down" />').appendTo(that.link);
            }
        },

        _click: function(e) {
            var that = this,
                element = that.element,
                field = element.data(FIELD),
                dir = element.data(DIR),
                options = that.options,
                sort = that.dataSource.sort() || [],
                idx,
                length;

            if (dir === ASC) {
                dir = DESC;
            } else if (dir === DESC && options.allowUnsort) {
                dir = undefined;
            } else {
                dir = ASC;
            }

            if (options.mode === SINGLE) {
                sort = [ { field: field, dir: dir } ];
            } else if (options.mode === "multiple") {
                for (idx = 0, length = sort.length; idx < length; idx++) {
                    if (sort[idx].field === field) {
                        sort.splice(idx, 1);
                        break;
                    }
                }
                sort.push({ field: field, dir: dir });
            }

            e.preventDefault();

            that.dataSource.sort(sort);
        }
    });

    kendo.ui.plugin(Sortable);
})(jQuery);
/*
* Kendo UI Web v2012.1.322 (http://kendoui.com)
* Copyright 2012 Telerik AD. All rights reserved.
*
* Kendo UI Web commercial licenses may be obtained at http://kendoui.com/web-license
* If you do not own a commercial license, this file shall be governed by the
* GNU General Public License (GPL) version 3.
* For GPL requirements, please review: http://www.gnu.org/copyleft/gpl.html
*/
(function($, undefined) {
    var kendo = window.kendo,
        ui = kendo.ui,
        DataSource = kendo.data.DataSource,
        Groupable = ui.Groupable,
        tbodySupportsInnerHtml = kendo.support.tbodyInnerHtml,
        Widget = ui.Widget,
        keys = kendo.keys,
        isPlainObject = $.isPlainObject,
        extend = $.extend,
        map = $.map,
        isArray = $.isArray,
        proxy = $.proxy,
        isFunction = $.isFunction,
        isEmptyObject = $.isEmptyObject,
        math = Math,
        REQUESTSTART = "requestStart",
        ERROR = "error",
        ROW_SELECTOR = "tbody>tr:not(.k-grouping-row,.k-detail-row):visible",
        DATA_CELL = ":not(.k-group-cell,.k-hierarchy-cell):visible",
        CELL_SELECTOR =  ROW_SELECTOR + ">td" + DATA_CELL,
        FIRST_CELL_SELECTOR = CELL_SELECTOR + ":first",
        EDIT = "edit",
        SAVE = "save",
        REMOVE = "remove",
        DETAILINIT = "detailInit",
        CHANGE = "change",
        SAVECHANGES = "saveChanges",
        MODELCHANGE = "modelChange",
        DATABOUND = "dataBound",
        DETAILEXPAND = "detailExpand",
        DETAILCOLLAPSE = "detailCollapse",
        FOCUSED = "k-state-focused",
        FOCUSABLE = "k-focusable",
        SELECTED = "k-state-selected",
        CLICK = "click",
        HEIGHT = "height",
        TABINDEX = "tabIndex",
        FUNCTION = "function",
        STRING = "string",
        DELETECONFIRM = "Are you sure you want to delete this record?",
        formatRegExp = /\}/ig,
        templateHashRegExp = /#/ig,
        COMMANDBUTTONTEMP = '<a class="k-button k-button-icontext #=className#" #=attr# href="\\#"><span class="#=iconClass# #=imageClass#"></span>#=text#</a>';

    var VirtualScrollable =  Widget.extend({
        init: function(element, options) {
            var that = this;

            Widget.fn.init.call(that, element, options);
            that.dataSource = options.dataSource;
            that.dataSource.bind(CHANGE, proxy(that.refresh, that));
            that.wrap();
        },

        options: {
            name: "VirtualScrollable",
            itemHeight: $.noop
        },

        wrap: function() {
            var that = this,
                // workaround for IE issue where scroll is not raised if container is same width as the scrollbar
                scrollbar = kendo.support.scrollbar() + 1,
                element = that.element;

            element.css( {
                width: "auto",
                paddingRight: scrollbar,
                overflow: "hidden"
            });
            that.content = element.children().first();
            that.wrapper = that.content.wrap('<div class="k-virtual-scrollable-wrap"/>')
                                .parent()
                                .bind("DOMMouseScroll", proxy(that._wheelScroll, that))
                                .bind("mousewheel", proxy(that._wheelScroll, that));

            if (kendo.support.touch) {
                that.drag = new kendo.Drag(that.wrapper, {
                    global: true,
                    move: function(e) {
                        that.verticalScrollbar.scrollTop(that.verticalScrollbar.scrollTop() - e.y.delta);
                        e.preventDefault();
                    }
                });
            }

            that.verticalScrollbar = $('<div class="k-scrollbar k-scrollbar-vertical" />')
                                        .css({
                                            width: scrollbar
                                        }).appendTo(element)
                                        .bind("scroll", proxy(that._scroll, that));
        },

        _wheelScroll: function(e) {
            var that = this,
                scrollTop = that.verticalScrollbar.scrollTop(),
                originalEvent = e.originalEvent,
                delta;

            e.preventDefault();

            if (originalEvent.wheelDelta) {
                delta = originalEvent.wheelDelta;
            } else if (originalEvent.detail) {
                delta = (-originalEvent.detail) * 10;
            } else if ($.browser.opera) {
                delta = -originalEvent.wheelDelta;
            }
            that.verticalScrollbar.scrollTop(scrollTop + (-delta));
        },

        _scroll: function(e) {
            var that = this,
                scrollTop = e.currentTarget.scrollTop,
                dataSource = that.dataSource,
                rowHeight = that.itemHeight,
                skip = dataSource.skip() || 0,
                start = that._rangeStart || skip,
                height = that.element.innerHeight(),
                isScrollingUp = !!(that._scrollbarTop && that._scrollbarTop > scrollTop),
                firstItemIndex = math.max(math.floor(scrollTop / rowHeight), 0),
                lastItemIndex = math.max(firstItemIndex + math.floor(height / rowHeight), 0);

            that._scrollTop = scrollTop - (start * rowHeight);
            that._scrollbarTop = scrollTop;

            if (!that._fetch(firstItemIndex, lastItemIndex, isScrollingUp)) {
                that.wrapper[0].scrollTop = that._scrollTop;
            }
        },

        _fetch: function(firstItemIndex, lastItemIndex, scrollingUp) {
            var that = this,
                dataSource = that.dataSource,
                itemHeight = that.itemHeight,
                take = dataSource.take(),
                rangeStart = that._rangeStart || dataSource.skip() || 0,
                currentSkip = math.floor(firstItemIndex / take) * take,
                fetching = false,
                prefetchAt = 0.33;

            if (firstItemIndex < rangeStart) {

                fetching = true;
                rangeStart = math.max(0, lastItemIndex - take);
                that._scrollTop = (firstItemIndex - rangeStart) * itemHeight;
                that._page(rangeStart, take);

            } else if (lastItemIndex >= rangeStart + take && !scrollingUp) {

                fetching = true;
                rangeStart = firstItemIndex;
                that._scrollTop = itemHeight;
                that._page(rangeStart, take);

            } else if (!that._fetching) {

                if (firstItemIndex < (currentSkip + take) - take * prefetchAt && firstItemIndex > take) {
                    dataSource.prefetch(currentSkip - take, take);
                }
                if (lastItemIndex > currentSkip + take * prefetchAt) {
                    dataSource.prefetch(currentSkip + take, take);
                }

            }
            return fetching;
        },

        _page: function(skip, take) {
            var that = this,
                dataSource = that.dataSource;

            clearTimeout(that._timeout);
            that._fetching = true;
            that._rangeStart = skip;

            if (dataSource.inRange(skip, take)) {
                dataSource.range(skip, take);
            } else {
                kendo.ui.progress(that.wrapper.parent(), true);
                that._timeout = setTimeout(function() {
                    dataSource.range(skip, take);
                }, 100);
            }
        },

        refresh: function() {
            var that = this,
                html = "",
                maxHeight = 250000,
                dataSource = that.dataSource,
                rangeStart = that._rangeStart,
                scrollbar = kendo.support.scrollbar(),
                wrapperElement = that.wrapper[0],
                totalHeight,
                idx,
                itemHeight;

            kendo.ui.progress(that.wrapper.parent(), false);
            clearTimeout(that._timeout);

            itemHeight = that.itemHeight = that.options.itemHeight() || 0;

            var addScrollBarHeight = (wrapperElement.scrollWidth > wrapperElement.offsetWidth) ? scrollbar : 0;

            totalHeight = dataSource.total() * itemHeight + addScrollBarHeight;

            for (idx = 0; idx < math.floor(totalHeight / maxHeight); idx++) {
                html += '<div style="width:1px;height:' + maxHeight + 'px"></div>';
            }

            if (totalHeight % maxHeight) {
                html += '<div style="width:1px;height:' + (totalHeight % maxHeight) + 'px"></div>';
            }

            that.verticalScrollbar.html(html);
            wrapperElement.scrollTop = that._scrollTop;

            if (that.drag) {
                that.drag.cancel();
            }

            if (rangeStart && !that._fetching) { // we are rebound from outside local range should be reset
                that._rangeStart = dataSource.skip();
            }
            that._fetching = false;
        }
    });

    function groupCells(count) {
        return new Array(count + 1).join('<td class="k-group-cell"></td>');
    }

    var defaultCommands = {
        create: {
            text: "Add new record",
            imageClass: "k-add",
            className: "k-grid-add",
            iconClass: "k-icon"
        },
        cancel: {
            text: "Cancel changes",
            imageClass: "k-cancel",
            className: "k-grid-cancel-changes",
            iconClass: "k-icon"
        },
        save: {
            text: "Save changes",
            imageClass: "k-update",
            className: "k-grid-save-changes",
            iconClass: "k-icon"
        },
        destroy: {
            text: "Delete",
            imageClass: "k-delete",
            className: "k-grid-delete",
            iconClass: "k-icon"
        },
        edit: {
            text: "Edit",
            imageClass: "k-edit",
            className: "k-grid-edit",
            iconClass: "k-icon"
        },
        update: {
            text: "Update",
            imageClass: "k-update",
            className: "k-grid-update",
            iconClass: "k-icon"
        },
        canceledit: {
            text: "Cancel",
            imageClass: "k-cancel",
            className: "k-grid-cancel",
            iconClass: "k-icon"
        }
    }

    /**
     *  @name kendo.ui.Grid.Description
     *
     *  @section
     *  <p>
     *      The Grid widget displays tabular data and offers rich support interacting with data,
     *      including paging, sorting, grouping, and selection. Grid is a powerful widget with
     *      many configuration options. It can be bound to local JSON data or to remote data
     *      using the Kendo DataSource component.
     *  </p>
     *  <h3>Getting Started</h3>
     *  There are two primary ways to create a Kendo Grid:
     *
     *  <ol>
     *      <li>From an existing HTML table element, defining columns, rows, and data in HTML</li>
     *      <li>From an HTML div element, defining columns and rows with configuration, and binding to data</li>
     *  </ol>
     *
     *  @exampleTitle Creating a <b>Grid</b> from existing HTML Table element
     *  @example
     *  <!-- Define the HTML table, with rows, columns, and data -->
     *  <table id="grid">
     *   <thead>
     *       <tr>
     *           <th data-field="title">Title</th>
     *           <th data-field="year">Year</th>
     *       </tr>
     *   </thead>
     *   <tbody>
     *       <tr>
     *           <td>Star Wars: A New Hope</td>
     *           <td>1977</td>
     *       </tr>
     *       <tr>
     *           <td>Star Wars: The Empire Strikes Back</td>
     *           <td>1980</td>
     *       </tr>
     *   </tbody>
     *  </table>
     *
     *  @exampleTitle Initialize the Kendo Grid
     *  @example
     *   $(document).ready(function(){
     *       $("#grid").kendoGrid();
     *   });
     *
     *  @exampleTitle Creating a <b>Grid</b> from existing HTML Div element
     *  @example
     *  <!-- Define the HTML div that will hold the Grid -->
     *  <div id="grid">
     *  </div>
     *
     *  @exampleTitle Initialize the Kendo Grid and configure columns & data binding
     *  @example
     *    $(document).ready(function(){
     *       $("#grid").kendoGrid({
     *           columns:[
     *               {
     *                   field: "FirstName",
     *                   title: "First Name"
     *               },
     *               {
     *                   field: "LastName",
     *                   title: "Last Name"
     *           }],
     *           dataSource: {
     *               data: [
     *                   {
     *                       FirstName: "Joe",
     *                       LastName: "Smith"
     *                   },
     *                   {
     *                       FirstName: "Jane",
     *                       LastName: "Smith"
     *               }]
     *           }
     *       });
     *   });
     *
     *  @section <h3>Configuring Grid Behavior</h3>
     *  Kendo Grid supports paging, sorting, grouping, and scrolling. Configuring any of
     *  these Grid behaviors is done using simple boolean configuration options. For
     *  example, the follow snippet shows how to enable all of these behaviors.
     *
     *  @exampleTitle Enabling Grid paging, sorting, grouping, and scrolling
     *  @example
     *    $(document).ready(function(){
     *       $("#grid").kendoGrid({
     *          groupable: true,
     *          scrollable: true,
     *          sortable: true,
     *          pageable: true
     *       });
     *   });
     *  @section
     *  By default, paging, grouping, and sorting are <strong>disabled</strong>. Scrolling is enabled by default.
     *
     *  <h3>Performance with Virtual Scrolling</h3>
     *  When binding to large data sets or when using large page sizes, reducing active in-memory
     *  DOM objects is important for performance. Kendo Grid provides built-in UI virtualization
     *  for highly optimized binding to large data sets. Enabling UI virtualization is done via simple configuration.
     *
     *  @exampleTitle Enabling Grid UI virtualization
     *  @example
     *    $(document).ready(function(){
     *       $("#grid").kendoGrid({
     *          scrollable: {
     *              virtual: true
     *          }
     *       });
     *   });
     *
     * @section
     * <h3>Accessing an Existing Grid</h3>
     * <p>
     *  You can reference an existing <b>Grid</b> instance via
     *  <a href="http://api.jquery.com/jQuery.data/">jQuery.data()</a>.
     *  Once a reference has been established, you can use the API to control
     *  its behavior.
     * </p>
     *
     * @exampleTitle Accessing an existing Grid instance
     * @example
     * var grid = $("#grid").data("kendoGrid");
     *
     */
    var Grid = Widget.extend(/** @lends kendo.ui.Grid.prototype */ {
        /**
         * @constructs
         * @extends kendo.ui.Widget
         * @param {DomElement} element DOM element
         * @param {Object} options Configuration options.
         * @option {kendo.data.DataSource | Object} [dataSource] Instance of DataSource or Object with DataSource configuration.
         * _exampleTitle Bind to a DataSource instance
         * _example
         * var sharedDataSource = new kendo.data.DataSource({
         *      data: [{title: "Star Wars: A New Hope", year: 1977}, {title: "Star Wars: The Empire Strikes Back", year: 1980}],
         *      pageSize: 1
         * });
         *
         * $("#grid").kendoGrid({
         *      dataSource: sharedDataSource
         *  });
         * _exampleTitle Bind to a local array
         * _example
         *  $("#grid").kendoGrid({
         *      dataSource: {
         *          data: [{title: "Star Wars: A New Hope", year: 1977}, {title: "Star Wars: The Empire Strikes Back", year: 1980}],
         *          pageSize: 1
         *      }
         *  });
         * @option {Function} [detailTemplate] Template to be used for rendering the detail rows in the grid.
         * See the <a href="http://demos.kendoui.com/web/grid/detailtemplate.html"><b>Detail Template</b></a> example.
         * @option {Object} [sortable] Defines whether grid columns are sortable.
         * _example
         * $("#grid").kendoGrid({
         *     sortable: true
         * });
         * //
         * // or
         * //
         * $("#grid").kendoGrid({
         *     sortable: {
         *         mode: "multiple", // enables multi-column sorting
         *         allowUnsort: true
         * });
         * @option {String} [sortable.mode] <"single"> Defines sorting mode. Possible values:
         * <div class="details-list">
         *    <dl>
         *         <dt>
         *              <b>"single"</b>
         *         </dt>
         *         <dd>
         *             Defines that only once column can be sorted at a time.
         *         </dd>
         *         <dt>
         *              <b>"multiple"</b>
         *         </dt>
         *         <dd>
         *              Defines that multiple columns can be sorted at a time.
         *         </dd>
         *    </dl>
         * </div>
         * @option {Boolean} [sortable.allowUnsort] <false>  Defines whether column can have unsorted state.
         * @option {Array} [columns] A collection of column objects or collection of strings that represents the name of the fields.
         * _example
         * var sharedDataSource = new kendo.data.DataSource({
         *      data: [{title: "Star Wars: A New Hope", year: 1977}, {title: "Star Wars: The Empire Strikes Back", year: 1980}],
         *      pageSize: 1
         * });
         * $("#grid").kendoGrid({
         *     dataSource: sharedDataSource,
         *     columns: [ { title: "Action", command: "destroy" }, // creates a column with delete buttons
         *                { title: "Title", field: "title", width: 200, template: "&lt;h1 id='title'&gt;${ title }&lt;/div&gt;" },
         *                { title: "Year", field: "year", filterable: false, sortable: true, format: "{0:dd/MMMM/yyyy}" } ];
         * });
         * @option {String} [columns.field] The field from the datasource that will be displayed in the column.
         * @option {String} [columns.title] The text that will be displayed in the column header.
         * @option {String} [columns.format] The format that will be applied on the column cells.
         * _example
         *  $("#grid").kendoGrid({
         *      dataSource: {
         *          data: createRandomData(50),
         *          pageSize: 10
         *      },
         *      columns: [
         *          {
         *              field: "BirthDate",
         *              title: "Birth Date",
         *              format: "{0:dd/MMMM/yyyy}"
         *         }
         *      ]
         *   });
         * @option {Boolean} [columns.filterable] <true> Specifies whether given column is filterable.
         * @option {Boolean} [columns.sortable] <true> Specifies whether given column is sortable.
         * @option {Function} [columns.editor] Provides a way to specify custom editor for this column.
         * _example
         * $(".k-grid").kendoGrid({
         *      dataSource: {
         *          data: createRandomData(50)
         *      },
         *      editable: true,
         *      columns: [
         *          {
         *              field: "Name",
         *              editor: function(container, options) {
         *                  // create a KendoUI AutoComplete widget as column editor
         *                   $('&lt;input name="' + options.field + '"/&gt;')
         *                       .appendTo(container)
         *                       .kendoAutoComplete({
         *                           dataTextField: "ProductName",
         *                           dataSource: {
         *                               transport: {
         *                                 //...
         *                               }
         *                           }
         *                       });
         *              }
         *          }
         *      ]
         *   });
         * @option {Object} [columns.editor.container] The container in which the editor must be added.
         * @option {Object} [columns.editor.options] Additional options.
         * @option {String} [columns.editor.options.field] The field for the editor.
         * @option {Object} [columns.editor.options.model] The model for the editor.
         * @option {String} [columns.width] The width of the column.
         * @option {Boolean} [columns.encoded] <true> Specified whether the column content is escaped. Disable encoding if the data contains HTML markup.
         * @option {String} [columns.command] Definition of command column. The supported built-in commands are: "create", "cancel", "save", "destroy".
         * @option {String} [columns.template] The template for column's cells.
         * _example
         *  $("#grid").kendoGrid({
         *      dataSource: {
         *          data: createRandomData(50),
         *          pageSize: 10
         *      },
         *      columns: [
         *          {
         *              field: "Name"
         *          },
         *          {
         *              field: "BirthDate",
         *              title: "Birth Date",
         *              template: '#= kendo.toString(BirthDate,"dd MMMM yyyy") #'
         *         }
         *      ]
         *   });
         * @option {Array} [toolbar] This is a list of commands for which the corresponding buttons will be rendered.
         * The supported built-in commands are: "create", "cancel", "save", "destroy".
         * _example
         *  $("#grid").kendoGrid({
         *      dataSource: {
         *          data: createRandomData(50),
         *          pageSize: 10
         *      },
         *      columns: [
         *          {
         *              field: "Name"
         *          },
         *          {
         *              field: "BirthDate",
         *              title: "Birth Date",
         *              template: '#= kendo.toString(BirthDate,"dd MMMM yyyy") #'
         *         }
         *      ]
         *      toolbar: [
         *          "create",
         *          { name: "save", text: "Save This Record" },
         *          { name: "cancel", template: '&lt;img src="icons/cancel.png' rel='cancel' /&gt;" }
         *      ],
         *      editable: true
         *   });
         * @option {String} [toolbar.name] The name of the command. One of the predefined or a custom.
         * @option {String} [toolbar.text] The text of the command that will be set on the button.
         * @option {String} [toolbar.template] The template for the command button.
         *
         * @option {Object} [editable] Indicates whether editing is enabled/disabled.
         * _example
         *  $("#grid").kendoGrid({
         *      dataSource: {
         *          data: createRandomData(50),
         *          pageSize: 10
         *      },
         *      columns: [
         *          {
         *              field: "Name"
         *          },
         *          {
         *              field: "BirthDate",
         *              title: "Birth Date",
         *              template: '#= kendo.toString(BirthDate,"dd MMMM yyyy") #'
         *         }
         *      ]
         *      toolbar: [
         *          "create",
         *          { name: "save", text: "Save This Record" },
         *          { name: "cancel", template: "&lt;img src="icons/cancel.png' rel='cancel' /&gt;" }
         *      ],
         *      editable: {
         *          update: "true", // puts the row in edit mode when it is clicked
         *          destroy: "false", // does not remove the row when it is deleted, but marks it for deletion
         *          confirmation: "Are you sure you want to remove this item?"
         *      }
         *  });
         * @option {String} [editable.mode] Indicates which of the available edit modes(incell(default)/inline/popup) will be used
         * @option {Boolean} [editable.update] Indicates whether item should be switched to edit mode on click.
         * @option {Boolean} [editable.destroy] Indicates whether item should be deleted when click on delete button.
         * @option {Boolean} [editable.confirmation] Defines the text that will be used in confirmation box when delete an item.
         * @option {Boolean} [editable.template] Template which will be use during popup editing
         * @option {Boolean} [pageable] <false> Indicates whether paging is enabled/disabled.
         * _example
         *  $("#grid").kendoGrid({
         *      dataSource: {
         *          data: createRandomData(50),
         *          pageSize: 10
         *      },
         *      columns: [
         *          {
         *              field: "Name"
         *          },
         *          {
         *              field: "BirthDate",
         *              title: "Birth Date",
         *              template: '#= kendo.toString(BirthDate,"dd MMMM yyyy") #'
         *         }
         *      ],
         *      pageable: true
         *  });
         * @option {Boolean} [groupable] <false> Indicates whether grouping is enabled/disabled.
         * _example
         *  $("#grid").kendoGrid({
         *      dataSource: {
         *          data: createRandomData(50),
         *          pageSize: 10
         *      },
         *      columns: [
         *          {
         *              field: "Name"
         *          },
         *          {
         *              field: "BirthDate",
         *              title: "Birth Date",
         *              template: '#= kendo.toString(BirthDate,"dd MMMM yyyy") #'
         *         }
         *      ],
         *      groupable: true
         *  });
         * @option {Boolean} [navigatable] <false> Indicates whether keyboard navigation is enabled/disabled.
         * _example
         *  $("#grid").kendoGrid({
         *      dataSource: {
         *          data: createRandomData(50),
         *          pageSize: 10
         *      },
         *      columns: [
         *          {
         *              field: "Name"
         *          },
         *          {
         *              field: "BirthDate",
         *              title: "Birth Date",
         *              template: '#= kendo.toString(BirthDate,"dd MMMM yyyy") #'
         *         }
         *      ],
         *      navigatable: true
         *  });
         * @option {String} [selectable] <undefined> Indicates whether selection is enabled/disabled. Possible values:
         * <div class="details-list">
         *    <dl>
         *         <dt>
         *              <b>"row"</b>
         *         </dt>
         *         <dd>
         *              Single row selection.
         *         </dd>
         *         <dt>
         *              <b>"cell"</b>
         *         </dt>
         *         <dd>
         *              Single cell selection.
         *         </dd>
         *         <dt>
         *              <b>"multiple, row"</b>
         *         </dt>
         *         <dd>
         *              Multiple row selection.
         *         </dd>
         *         <dt>
         *              <b>"multiple, cell"</b>
         *         </dt>
         *         <dd>
         *              Multiple cell selection.
         *         </dd>
         *    </dl>
         * </div>
         * @option {Boolean} [autoBind] <true> Indicates whether the grid will call read on the DataSource initially.
         * _example
         *  $("#grid").kendoGrid({
         *      dataSource: sharedDataSource,
         *      columns: [
         *          {
         *              field: "Name"
         *          },
         *          {
         *              field: "BirthDate",
         *              title: "Birth Date",
         *              template: '#= kendo.toString(BirthDate,"dd MMMM yyyy") #'
         *         }
         *      ],
         *      autoBind: false // the grid will not be populated with data until read() is called on the sharedDataSource
         *  });
         * @option {Boolean | Object} [scrollable] <true> Enable/disable grid scrolling. Possible values:
         * <div class="details-list">
         *    <dl>
         *         <dt>
         *              <b>true</b>
         *         </dt>
         *         <dd>
         *              Enables grid vertical scrolling
         *         </dd>
         *         <dt>
         *              <b>false</b>
         *         </dt>
         *         <dd>
         *              Disables grid vertical scrolling
         *         </dd>
         *         <dt>
         *              <b>{ virtual: false }</b>
         *         </dt>
         *         <dd>
         *              Enables grid vertical scrolling without data virtualization. Same as first option.
         *         </dd>
         *         <dt>
         *              <b>{ virtual: true }</b>
         *         </dt>
         *         <dd>
         *              Enables grid vertical scrolling with data virtualization.
         *         </dd>
         *    </dl>
         * </div>
         * _example
         *  $("#grid").kendoGrid({
         *      scrollable: {
         *          virtual: true //false
         *      }
         *  });
         * @option {Function} [rowTemplate] Template to be used for rendering the rows in the grid.
         * _example
         *  //template
         *  &lt;script id="rowTemplate" type="text/x-kendo-tmpl"&gt;
         *      &lt;tr&gt;
         *          &lt;td&gt;
         *              &lt;img src="${ BoxArt.SmallUrl }" alt="${ Name }" /&gt;
         *          &lt;/td&gt;
         *          &lt;td&gt;
         *              ${ Name }
         *          &lt;/td&gt;
         *          &lt;td&gt;
         *              ${ AverageRating }
         *          &lt;/td&gt;
         *      &lt;/tr&gt;
         *  &lt;/script&gt;
         *
         *  //grid intialization
         *  &lt;script&gt;PO details informaiton
         *      $("#grid").kendoGrid({
         *          dataSource: dataSource,
         *          rowTemplate: kendo.template($("#rowTemplate").html()),
         *          height: 200
         *      });
         *  &lt;/script&gt;
         */
        init: function(element, options) {
            var that = this;

            options = isArray(options) ? { dataSource: options } : options;

            Widget.fn.init.call(that, element, options);

            that._element();

            that._columns(that.options.columns);

            that._dataSource();

            that._tbody();

            that._pageable();

            that._groupable();

            that._toolbar();

            that._thead();

            that._templates();

            that._navigatable();

            that._selectable();

            that._details();

            that._editable();

            if (that.options.autoBind) {
                that.dataSource.fetch();
            }

            kendo.notify(that);
        },

        events: [
            /**
             * Fires when the grid selection has changed.
             * @name kendo.ui.Grid#change
             * @event
             * @param {Event} e
             * @example
             *  $("#grid").kendoGrid({
             *      change: function(e) {
             *          // handle event
             *      }
             *  });
             *  @exampleTitle To set after initialization
             *  @example
             *  // get a reference to the grid
             *  var grid = $("#grid").data("kendoGrid");
             *  // bind to the change event
             *  grid.bind("change", function(e) {
             *      // handle event
             *  }
             */
            CHANGE,
            "dataBinding",
            /**
             * Fires when the grid has received data from the data source.
             * @name kendo.ui.Grid#dataBound
             * @event
             * @param {Event} e
             * @example
             *  $("#grid").kendoGrid({
             *      dataBound: function(e) {
             *          // handle event
             *      }
             *  });
             *  @exampleTitle To set after initialization
             *  @example
             *  // get a reference to the grid
             *  var grid = $("#grid").data("kendoGrid");
             *  // bind to the dataBound event
             *  grid.bind("dataBound", function(e) {
             *      // handle event
             *  }
             */
            DATABOUND,
            /**
             * Fires when the grid detail row is expanded.
             * @name kendo.ui.Grid#detailExpand
             * @event
             * @param {Event} e
             * @param {Object} e.masterRow The jQuery element representing master row.
             * @param {Object} e.detailRow The jQuery element representing detail row.
             * @example
             *  $("#grid").kendoGrid({
             *      detailExpand: function(e) {
             *          // handle event
             *      }
             *  });
             *  @exampleTitle To set after initialization
             *  @example
             *  // get a reference to the grid
             *  var grid = $("#grid").data("kendoGrid");
             *  // bind to the detailExpand event
             *  grid.bind("detailExpand", function(e) {
             *      // handle event
             *  }
             */
            DETAILEXPAND,
            /**
             * Fires when the grid detail row is collapsed.
             * @name kendo.ui.Grid#detailCollapse
             * @event
             * @param {Event} e
             * @param {Object} e.masterRow The jQuery element representing master row.
             * @param {Object} e.detailRow The jQuery element representing detail row.
             * @example
             *  $("#grid").kendoGrid({
             *      detailCollapse: function(e) {
             *          // handle event
             *      }
             *  });
             * @exampleTitle To set after initialization
             * @example
             *  // get a reference to the grid
             *  var grid = $("#grid").data("kendoGrid");
             *  // bind to the detailCollapse event
             *  grid.bind("detailCollapse", function(e) {
             *      // handle event
             *  }
             */
            DETAILCOLLAPSE,
            /**
             * Fires when the grid detail is initialized.
             * @name kendo.ui.Grid#detailInit
             * @event
             * @param {Event} e
             * @param {Object} e.masterRow The jQuery element representing master row.
             * @param {Object} e.detailRow The jQuery element representing detail row.
             * @param {Object} e.detailCell The jQuery element representing detail cell.
             * @param {Object} e.data The data for the master row.
             * @example
             *  $("#grid").kendoGrid({
             *      detailInit: function(e) {
             *          // handle event
             *  });
             * @exampleTitle To set after initialization
             * @example
             *  // get a reference to the grid
             *  var grid = $("#grid").data("kendoGrid");
             *  // bind to the detailInit event
             *  grid.bind("detailInit", function(e) {
             *      // handle event
             *  }
             */
            DETAILINIT,
            /**
             * Fires when the grid enters edit mode.
             * @name kendo.ui.Grid#edit
             * @event
             * @param {Event} e
             * @param {Object} e.container The jQuery element to be edited.
             * @param {Object} e.model The model to be edited.
             * @example
             *  $("#grid").kendoGrid({
             *      edit: function(e) {
             *          // handle event
             *  });
             * @exampleTitle To set after initialization
             * @example
             *  // get a reference to the grid
             *  var grid = $("#grid").data("kendoGrid");
             *  // bind to the edit event
             *  grid.bind("edit", function(e) {
             *      // handle event
             *  }
             */
            EDIT,
            /**
             * Fires before the grid item is changed.
             * @name kendo.ui.Grid#save
             * @event
             * @param {Event} e
             * @param {Object} e.values The values entered by the user.
             * @param {Object} e.container The jQuery element which is in edit mode.
             * @param {Object} e.model The edited model.
             * @example
             *  $("#grid").kendoGrid({
             *      save: function(e) {
             *          // handle event
             *  });
             * @exampleTitle To set after initialization
             * @example
             *  // get a reference to the grid
             *  var grid = $("#grid").data("kendoGrid");
             *  // bind to the save event
             *  grid.bind("save", function(e) {
             *      // handle event
             *  }
             */
            SAVE,
            /**
             * Fires before the grid item is removed.
             * @name kendo.ui.Grid#remove
             * @event
             * @param {Event} e
             * @param {Object} e.row The row element to be deleted.
             * @param {Object} e.model The model which to be deleted.
             * @example
             *  $("#grid").kendoGrid({
             *      remove: function(e) {
             *          // handle event
             *  });
             *  @exampleTitle To set after initialization
             *  @example
             *  // get a reference to the grid
             *  var grid = $("#grid").data("kendoGrid");
             *  // bind to the remove event
             *  grid.bind("remove", function(e) {
             *      // handle event
             *  }
             */
            REMOVE,
            /**
             * Fires before the grid calls DataSource sync.
             * @name kendo.ui.Grid#saveChanges
             * @event
             * @param {Event} e
             * @example
             *  $("#grid").kendoGrid({
             *      saveChanges: function(e) {
             *          // handle event
             *  });
             *  @exampleTitle To set after initialization
             *  @example
             *  // get a reference to the grid
             *  var grid = $("#grid").data("kendoGrid");
             *  // bind to the saveChanges event
             *  grid.bind("saveChanges", function(e) {
             *      // handle event
             *  }
             */
            SAVECHANGES
        ],

        setDataSource: function(dataSource) {
            var that = this;

            that.options.dataSource = dataSource;

            that._dataSource();

            that._pageable();

            that._groupable();

            that._thead();

            dataSource.fetch();
        },

        options: {
            name: "Grid",
            columns: [],
            toolbar: null,
            autoBind: true,
            scrollable: true,
            sortable: false,
            selectable: false,
            navigatable: false,
            pageable: false,
            editable: false,
            rowTemplate: "",
            groupable: false,
            dataSource: {}
        },

        setOptions: function(options) {
            var that = this;

            Widget.fn.setOptions.call(this, options);

            that._templates();
        },

        items: function() {
            return this.tbody.children(':not(.k-grouping-row,.k-detail-row)');
        },

        _element: function() {
            var that = this,
                table = that.element;

            if (!table.is("table")) {
                table = $("<table />").appendTo(that.element);
            }

            that.table = table.attr("cellspacing", 0);

            that._wrapper();
        },

        /**
         * Returns the index of the cell in the grid item skipping group and hierarchy cells.
         * @param {Selector | DOM Element} cell Target cell.
         * @example
         *  // get a reference to the grid widget
         *  var grid = $("#grid").data("kendoGrid");
         *  // get the index of the row
         *  // TODO: add specific function call here
         */
        cellIndex: function(td) {
            return $(td).parent().find('td:not(.k-group-cell,.k-hierarchy-cell)').index(td);
        },

        _modelForContainer: function(container) {
            var id = (container.is("tr") ? container : container.closest("tr")).attr(kendo.attr("uid"));

            return this.dataSource.getByUid(id);
        },

        _editable: function() {
            var that = this,
                cell,
                model,
                column,
                editable = that.options.editable,
                handler = function () {
                    var target = document.activeElement,
                        cell = that._editContainer;

                    if (cell && !$.contains(cell[0], target) && cell[0] !== target && !$(target).closest(".k-animation-container").length) {
                        if (that.editable.end()) {
                            that.closeCell();
                        }
                    }
                };

            if (editable) {
                var mode = that._editMode();

                if (mode === "incell") {
                    if (editable.update !== false) {
                        that.wrapper.delegate("tr:not(.k-grouping-row) > td:not(.k-hierarchy-cell,.k-detail-cell,.k-group-cell,.k-edit-cell,:has(a.k-grid-delete))", CLICK, function(e) {
                            var td = $(this)

                            if (td.closest("tbody")[0] !== that.tbody[0] || $(e.target).is(":input")) {
                                return;
                            }

                            if (that.editable) {
                                if (that.editable.end()) {
                                    that.closeCell();
                                    that.editCell(td);
                                }
                            } else {
                                that.editCell(td);
                            }

                        });

                        that.wrapper.bind("focusin", function(e) {
                            clearTimeout(that.timer);
                            that.timer = null;
                        });
                        that.wrapper.bind("focusout", function(e) {
                            that.timer = setTimeout(handler, 1);
                        });
                    }
                } else {
                    if (editable.update !== false) {
                        that.wrapper.delegate("tbody>tr:not(.k-detail-row,.k-grouping-row):visible a.k-grid-edit", CLICK, function(e) {
                            e.preventDefault();
                            that.editRow($(this).closest("tr"));
                        });
                    }
                }

                if (editable.destroy !== false) {
                    that.wrapper.delegate("tbody>tr:not(.k-detail-row,.k-grouping-row):visible a.k-grid-delete", CLICK, function(e) {
                        e.preventDefault();
                        that.removeRow($(this).closest("tr"));
                    });
               }
            }
        },

        /**
         * Puts the specified table cell in edit mode. It requires a jQuery object representing the cell. The editCell method triggers edit event.
         * @param {Selector} cell Cell to be edited.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // edit first table cell
         * grid.editCell(grid.tbody.find(">tr>td:first"));
         */
        editCell: function(cell) {
            var that = this,
                column = that.columns[that.cellIndex(cell)],
                model = that._modelForContainer(cell);


            if (model && (!model.editable || model.editable(column.field)) && !column.command && column.field) {

                that._attachModelChange(model);

                that._editContainer = cell;

                that.editable = cell.addClass("k-edit-cell")
                    .kendoEditable({
                        fields: { field: column.field, format: column.format, editor: column.editor },
                        model: model,
                        change: function(e) {
                            if (that.trigger(SAVE, { values: e.values, container: cell, model: model } )) {
                                e.preventDefault();
                            }
                        }
                    }).data("kendoEditable");

                cell.parent().addClass("k-grid-edit-row");

                that.trigger(EDIT, { container: cell, model: model });
            }
        },

        _distroyEditable: function() {
            var that = this;

            if (that.editable) {
                that._detachModelChange();

                that.editable.distroy();
                delete that.editable;

                if (that._editMode() === "popup") {
                    that._editContainer.data("kendoWindow").close();
                }

                that._editContainer = null;
            }
        },

        _attachModelChange: function(model) {
            var that = this;

            that._modelChangeHandler = function(e) {
                that._modelChange({ field: e.field, model: this });
            };

            model.bind("change", that._modelChangeHandler);
        },

        _detachModelChange: function() {
            var that = this,
                container = that._editContainer,
                model = that._modelForContainer(container);

            if (model) {
                model.unbind(CHANGE, that._modelChangeHandler);
            }
        },

        /**
         * Closes current edited cell.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // close the cell being edited
         * grid.closeCell();
         */
        closeCell: function() {
            var that = this,
                cell = that._editContainer.removeClass("k-edit-cell"),
                id = cell.closest("tr").attr(kendo.attr("uid")),
                column = that.columns[that.cellIndex(cell)],
                model = that.dataSource.getByUid(id);

            cell.parent().removeClass("k-grid-edit-row");

            that._distroyEditable(); // editable should be destoryed before content of the container is changed

            that._displayCell(cell, column, model);

            if (cell.hasClass("k-dirty-cell")) {
                $('<span class="k-dirty"/>').prependTo(cell);
            }
        },

        _displayCell: function(cell, column, dataItem) {
            var that = this,
                state = { storage: {}, count: 0 },
                settings = extend({}, kendo.Template, that.options.templateSettings),
                tmpl = kendo.template(that._cellTmpl(column, state), settings);

            if (state.count > 0) {
                tmpl = proxy(tmpl, state.storage);
            }

            cell.empty().html(tmpl(dataItem));
        },

        /**
         * Removes the specified row from the grid. The removeRow method triggers remove event.
         * (Note: In inline or popup edit modes the changes will be automatically synced)
         * @param {Selector | DOM Element} row Row to be removed.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // remove first table row
         * grid.removeRow(grid.tbody.find(">tr:first"));
         */
        removeRow: function(row) {
            var that = this,
                model,
                mode;

            if (!that._confirmation()) {
                return;
            }

            row = $(row).hide();
            model = that._modelForContainer(row);

            if (model && !that.trigger(REMOVE, { row: row, model: model })) {
                that.dataSource.remove(model);

                mode = that._editMode();

                if (mode === "inline" || mode === "popup") {
                    that.dataSource.sync();
                }
            }
        },

        _editMode: function() {
            var mode = "incell",
                editable = this.options.editable;

            if (editable !== true) {
                mode = editable.mode || editable;
            }

            return mode;
        },

        /**
         * Switches the specified row from the grid into edit mode. The editRow method triggers edit event.
         * @param {Selector | DOM Element} row Row to be edited.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // edit first table row
         * grid.editRow(grid.tbody.find(">tr:first"));
         */
        editRow: function(row) {
            var that = this,
                model = that._modelForContainer(row),
                container;

            that.cancelRow();

            if (model) {

                that._attachModelChange(model);

                if (that._editMode() === "popup") {
                    that._createPopUpEditor(model);
                } else {
                    that._createInlineEditor(row, model);
                }
                container = that._editContainer;

                container.delegate("a.k-grid-cancel", CLICK, function(e) {
                    e.preventDefault();
                    that.cancelRow();
                });

                container.delegate("a.k-grid-update", CLICK, function(e) {
                    e.preventDefault();
                    that.saveRow();
                });
            }
        },

        _createPopUpEditor: function(model) {
            var that = this,
                html = '<div><div class="k-edit-form-container">',
                column,
                fields = [],
                idx,
                length,
                tmpl,
                editable = that.options.editable,
                options = isPlainObject(editable) ? editable.window : {},
                settings = extend({}, kendo.Template, that.options.templateSettings);

            if (editable.template) {
                html += (kendo.template(editable.template, settings))(model);
            } else {
                for (idx = 0, length = that.columns.length; idx < length; idx++) {
                    column = that.columns[idx];

                    if (!column.command) {
                        html += '<div class="k-edit-label"><label for="' + column.field + '">' + (column.title || column.field || "") + '</label></div>';

                        if ((!model.editable || model.editable(column.field)) && column.field) {
                            fields.push({ field: column.field, format: column.format, editor: column.editor });
                            html += '<div ' + kendo.attr("container-for") + '="' + column.field + '" class="k-edit-field"></div>';
                        } else {
                            var state = { storage: {}, count: 0 };

                            tmpl = kendo.template(that._cellTmpl(column, state), settings);

                            if (state.count > 0) {
                                tmpl = proxy(tmpl, state.storage);
                            }

                            html += '<div class="k-edit-field">' + tmpl(model) + '</div>';
                        }
                    }
                }
            }

            html += that._createButton("update") + that._createButton("canceledit");
            html += '</div></div>';

            var container = that._editContainer = $(html)
                .appendTo(that.wrapper)
                .kendoWindow(extend({
                    modal: true,
                    resizable: false,
                    draggable: true,
                    title: "Edit",
                    visible: false
                }, options));

            var wnd = container.data("kendoWindow");

            wnd.wrapper.delegate(".k-close", "click", function() {
                    that.cancelRow();
                });

            that.editable = that._editContainer
                .kendoEditable({
                    fields: fields,
                    model: model,
                    clearContainer: false
                }).data("kendoEditable");

            wnd.center().open();

            that.trigger(EDIT, { container: container, model: model });
        },

        _createInlineEditor: function(row, model) {
            var that = this,
                column,
                cell,
                fields = [],
                idx,
                length;

            row.children(":not(.k-group-cell,.k-hierarchy-cell)").each(function() {
                cell = $(this);
                column = that.columns[that.cellIndex(cell)];

                if (!column.command && column.field && (!model.editable || model.editable(column.field))) {
                    fields.push({ field: column.field, format: column.format, editor: column.editor });
                    cell.attr("data-container-for", column.field);
                    cell.empty();
                } else if (column.command && hasCommand(column.command, "edit")) {
                    cell.empty();
                    $(that._createButton("update") + that._createButton("canceledit")).appendTo(cell);
                }
            });

            that._editContainer = row;

            that.editable = row
            .addClass("k-grid-edit-row")
            .kendoEditable({
                fields: fields,
                model: model,
                clearContainer: false
            }).data("kendoEditable");

            that.trigger(EDIT, { container: row, model: model });
        },

        /**
         * Switch the current edited row into dislay mode and revert changes made to the data
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * grid.cancelRow();
         */
        cancelRow: function() {
            var that = this,
                container = that._editContainer,
                newRow,
                model

            if (container) {
                model = that._modelForContainer(container);

                that.dataSource.cancelChanges(model);

                if (that._editMode() !== "popup") {
                    that._displayRow(container);
                }

                that._distroyEditable();
            }
        },

        /**
         * Switch the current edited row into dislay mode and save changes made to the data
         * (Note: the changes will be automatically synced)
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * grid.saveRow();
         */
        saveRow: function() {
            var that = this,
                container = that._editContainer,
                model = that._modelForContainer(container),
                editable = that.editable;

            if (container && editable && editable.end() &&
                !that.trigger(SAVE, { container: container, model: model } )) {

                if (that._editMode() !== "popup") {
                    that._displayRow(container);
                }

                that._distroyEditable();
                that.dataSource.sync();
            }
        },

        _displayRow: function(row) {
            var that = this,
                model = that._modelForContainer(row);

            if (model) {
                row.replaceWith($((row.hasClass("k-alt") ? that.altRowTemplate : that.rowTemplate)(model)));
            }
        },

        _showMessage: function(text) {
            return confirm(text);
        },

        _confirmation: function() {
            var that = this,
                editable = that.options.editable,
                confirmation = editable === true || typeof editable === STRING ? DELETECONFIRM : editable.confirmation;

            return confirmation !== false && confirmation != undefined ? that._showMessage(confirmation) : true;
        },

        /**
         * Cancels any pending changes during. Deleted rows are restored. Inserted rows are removed. Updated rows are restored to their original values.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * grid.cancelChanges();
         */
        cancelChanges: function() {
            this.dataSource.cancelChanges();
        },

        /**
         * Calls DataSource sync to submit any pending changes if state is valid. The saveChanges method triggers saveChanges event.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * grid.saveChanges();
         */
        saveChanges: function() {
            var that = this;

            if (((that.editable && that.editable.end()) || !that.editable) && !that.trigger(SAVECHANGES)) {
                that.dataSource.sync();
            }
        },

        /**
         * Adds a new empty table row in edit mode. The addRow method triggers edit event.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * grid.addRow();
         */
        addRow: function() {
            var that = this,
                options = that.options,
                index,
                dataSource = that.dataSource;

            if ((that.editable && that.editable.end()) || !that.editable) {
                index = dataSource.indexOf((dataSource.view() || [])[0]);
                if (index < 0) {
                    index = 0;
                }

                var model = dataSource.insert(index, {}),
                    id = model.uid,
                    mode = that._editMode(),
                    row = that.table.find("tr[" + kendo.attr("uid") + "=" + id + "]"),
                    cell = row.children("td:not(.k-group-cell,.k-hierarchy-cell)").first();

                if ((mode === "inline" || mode === "popup") && row.length) {
                    that.editRow(row);
                } else if (cell.length) {
                    that.editCell(cell);
                }
            }
        },

        _toolbar: function() {
            var that = this,
                wrapper = that.wrapper,
                toolbar = that.options.toolbar,
                template;

            if (toolbar) {
                toolbar = isFunction(toolbar) ? toolbar({}) : (typeof toolbar === STRING ? toolbar : that._toolbarTmpl(toolbar).replace(templateHashRegExp, "\\#"));

                template = proxy(kendo.template(toolbar), that)

                $('<div class="k-toolbar k-grid-toolbar" />')
                    .html(template({}))
                    .prependTo(wrapper)
                    .delegate(".k-grid-add", CLICK, function(e) { e.preventDefault(); that.addRow(); })
                    .delegate(".k-grid-cancel-changes", CLICK, function(e) { e.preventDefault(); that.cancelChanges(); })
                    .delegate(".k-grid-save-changes", CLICK, function(e) { e.preventDefault(); that.saveChanges(); });
            }
        },

        _toolbarTmpl: function(commands) {
            var that = this,
                idx,
                length,
                html = "",
                options,
                commandName,
                template,
                command;

            if (isArray(commands)) {
                for (idx = 0, length = commands.length; idx < length; idx++) {
                    html += that._createButton(commands[idx]);
                }
            }
            return html;
        },

        _createButton: function(command) {
            var that = this,
                template = command.template || COMMANDBUTTONTEMP,
                commandName = typeof command === STRING ? command : command.name,
                options = { className: "", text: commandName, imageClass: "", attr: "", iconClass: "" };

            if (isPlainObject(command)) {
                options = extend(true, options, defaultCommands[commandName], command);
            } else {
                options = extend(true, options, defaultCommands[commandName]);
            }

            return kendo.template(template)(options);
        },

        _groupable: function() {
            var that = this,
                wrapper = that.wrapper,
                groupable = that.options.groupable;

            if (!that.groupable) {
                that.table.delegate(".k-grouping-row .k-collapse, .k-grouping-row .k-expand", CLICK, function(e) {
                    var element = $(this),
                    group = element.closest("tr");

                    if(element.hasClass('k-collapse')) {
                        that.collapseGroup(group);
                    } else {
                        that.expandGroup(group);
                    }
                    e.preventDefault();
                    e.stopPropagation();
                });
            }

            if (groupable) {
                if(!wrapper.has("div.k-grouping-header")[0]) {
                    $("<div />").addClass("k-grouping-header").html("&nbsp;").prependTo(wrapper);
                }

                if (that.groupable) {
                    that.groupable.destroy();
                }

                that.groupable = new Groupable(wrapper, {
                    filter: "th:not(.k-group-cell)[" + kendo.attr("field") + "][" + kendo.attr("groupable") + "!=false]",
                    groupContainer: "div.k-grouping-header",
                    dataSource: that.dataSource
                });
            }
        },

        _selectable: function() {
            var that = this,
                multi,
                cell,
                selectable = that.options.selectable;

            if (selectable) {
                multi = typeof selectable === STRING && selectable.toLowerCase().indexOf("multiple") > -1;
                cell = typeof selectable === STRING && selectable.toLowerCase().indexOf("cell") > -1;

                that.selectable = new kendo.ui.Selectable(that.table, {
                    filter: ">" + (cell ? CELL_SELECTOR : ROW_SELECTOR),
                    multiple: multi,
                    change: function() {
                        that.trigger(CHANGE);
                    }
                });

                if (that.options.navigatable) {
                    that.wrapper.keydown(function(e) {
                        var current = that.current();
                        if (e.keyCode === keys.SPACEBAR && e.target == that.wrapper[0] && !current.hasClass("k-edit-cell")) {
                            e.preventDefault();
                            e.stopPropagation();
                            current = cell ? current : current.parent();

                            if(multi) {
                                if(!e.ctrlKey) {
                                    that.selectable.clear();
                                } else {
                                    if(current.hasClass(SELECTED)) {
                                        current.removeClass(SELECTED);
                                        current = null;
                                    }
                                }
                            } else {
                                that.selectable.clear();
                            }

                            that.selectable.value(current);
                        }
                    });
                }
            }
        },

        /**
         * Clears currently selected items.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // clear the selection of items in the grid
         * grid.clearSelection();
         */
        clearSelection: function() {
            var that = this;
            that.selectable.clear();
            that.trigger(CHANGE);
        },

        /**
         * Selects the specified Grid rows/cells. If called without arguments - returns the selected rows/cells.
         * @param {Selector | Array} items Items to select.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // selects first grid item
         * grid.select(grid.tbody.find(">tr:first"));
         */
        select: function(items) {
            var that = this,
                selectable = that.selectable;

            items = $(items);
            if(items.length) {
                if(!selectable.options.multiple) {
                    selectable.clear();
                    items = items.first();
                }
                selectable.value(items);
                return;
            }

            return selectable.value();
        },

        current: function(element) {
            var that = this,
                current = that._current;

            if (element !== undefined && element.length) {
                if (!current || current[0] !== element[0]) {
                    element.addClass(FOCUSED);
                    if (current) {
                        current.removeClass(FOCUSED);
                    }
                    that._current = element;
                    that._scrollTo(element.parent()[0]);
                }
            }

            return that._current;
        },

        _scrollTo: function(element) {
            if(!element || !this.options.scrollable) {
                return;
            }

            var elementOffsetTop = element.offsetTop,
                container = this.content[0],
                elementOffsetHeight = element.offsetHeight,
                containerScrollTop = container.scrollTop,
                containerOffsetHeight = container.clientHeight,
                bottomDistance = elementOffsetTop + elementOffsetHeight;

            container.scrollTop = containerScrollTop > elementOffsetTop
                                    ? elementOffsetTop
                                    : bottomDistance > (containerScrollTop + containerOffsetHeight)
                                    ? bottomDistance - containerOffsetHeight
                                    : containerScrollTop;
        },

        _navigatable: function() {
            var that = this,
                wrapper = that.wrapper,
                table = that.table.addClass(FOCUSABLE),
                currentProxy = proxy(that.current, that),
                selector = "." + FOCUSABLE + " " + CELL_SELECTOR,
                browser = $.browser,
                clickCallback = function(e) {
                    var currentTarget = $(e.currentTarget);
                    if (currentTarget.closest("tbody")[0] !== that.tbody[0]) {
                        return;
                    }

                    currentProxy(currentTarget);
                    if(!$(e.target).is(":button,a,:input,a>.k-icon,textarea,span.k-icon,.k-input")) {
                        setTimeout(function() { wrapper.focus(); } );
                    }

                    e.stopPropagation();
                };

            if (that.options.navigatable) {
                wrapper.bind({
                    focus: function() {
                        var current = that._current;
                        if(current && current.is(":visible")) {
                            current.addClass(FOCUSED);
                        } else {
                            currentProxy(that.table.find(FIRST_CELL_SELECTOR));
                        }
                    },
                    focusout: function(e) {
                        if (that._current) {
                            that._current.removeClass(FOCUSED);
                        }
                        e.stopPropagation();
                    },
                    keydown: function(e) {
                        var key = e.keyCode,
                            current = that.current(),
                            shiftKey = e.shiftKey,
                            dataSource = that.dataSource,
                            pageable = that.options.pageable,
                            canHandle = !$(e.target).is(":button,a,:input,a>.t-icon"),
                            isInCell = that._editMode() == "incell",
                            currentIndex,
                            cell,
                            handled = false;

                        if (canHandle && keys.UP === key) {
                            currentProxy(current ? current.parent().prevAll(ROW_SELECTOR).last().children(":eq(" + current.index() + "),:eq(0)").last() : table.find(FIRST_CELL_SELECTOR));
                            handled = true;
                        } else if (canHandle && keys.DOWN === key) {
                            currentProxy(current ? current.parent().nextAll(ROW_SELECTOR).first().children(":eq(" + current.index() + "),:eq(0)").last() : table.find(FIRST_CELL_SELECTOR));
                            handled = true;
                        } else if (canHandle && keys.LEFT === key) {
                            currentProxy(current ? current.prevAll(DATA_CELL + ":first") : table.find(FIRST_CELL_SELECTOR));
                            handled = true;
                        } else if (canHandle && keys.RIGHT === key) {
                            currentProxy(current ? current.nextAll(":visible:first") : table.find(FIRST_CELL_SELECTOR));
                            handled = true;
                        } else if (canHandle && pageable && keys.PAGEDOWN == key) {
                            that._current = null;
                            dataSource.page(dataSource.page() + 1);
                            handled = true;
                        } else if (canHandle && pageable && keys.PAGEUP == key) {
                            that._current = null;
                            dataSource.page(dataSource.page() - 1);
                            handled = true;
                        } else if (that.options.editable) {
                            current = current ? current : table.find(FIRST_CELL_SELECTOR);
                            if (keys.ENTER == key || keys.F12 == key) {
                                that._handleEditing(current);
                                handled = true;
                            } else if (keys.TAB == key && isInCell) {
                                cell = shiftKey ? current.prevAll(DATA_CELL + ":first") : current.nextAll(":visible:first");
                                if (!cell.length) {
                                    cell = current.parent()[shiftKey ? "prevAll" : "nextAll"]("tr:not(.k-grouping-row,.k-detail-row):visible")
                                        .children(DATA_CELL + (shiftKey ? ":last" : ":first"));
                                }

                                if (cell.length) {
                                    that._handleEditing(current, cell);
                                    handled = true;
                                }
                            } else if (keys.ESC == key && that._editContainer) {
                                if (that._editContainer.has(current[0]) || current[0] === that._editContainer[0]) {
                                    if (isInCell) {
                                        that.closeCell();
                                    } else {
                                        currentIndex = that.items().index(current.parent());
                                        document.activeElement.blur();
                                        that.cancelRow();
                                        if (currentIndex >= 0) {
                                            that.current(that.items().eq(currentIndex).children().filter(DATA_CELL).first());
                                        }
                                    }

                                    if (browser.msie && parseInt(browser.version) < 9) {
                                        document.body.focus();
                                    }
                                    wrapper.focus();
                                    handled = true;
                                }
                            }
                        }

                        if(handled) {
                            e.preventDefault();
                            e.stopPropagation();
                        }
                    }
                });

                wrapper.delegate(selector, "mousedown", clickCallback); }
        },

        _handleEditing: function(current, next) {
            var that = this,
                mode = that._editMode(),
                editContainer = that._editContainer,
                idx,
                isEdited;

            if (mode == "incell") {
                isEdited = current.hasClass("k-edit-cell");
            } else {
                isEdited = current.parent().hasClass("k-grid-edit-row");
            }

            if (that.editable) {
                if ($.contains(editContainer[0], document.activeElement)) {
                    document.activeElement.blur();
                }

                if (that.editable.end()) {
                    if (mode == "incell") {
                        that.closeCell();
                    } else {
                        if (current.parent()[0] === editContainer[0]) {
                            idx = that.items().index(current.parent());
                        } else {
                            idx = that.items().index(editContainer);
                        }
                        that.saveRow();
                        that.current(that.items().eq(idx).children().filter(DATA_CELL).first());
                        isEdited = true;
                    }
                } else {
                    if (mode == "incell") {
                        that.current(editContainer);
                    } else {
                        that.current(editContainer.children().filter(DATA_CELL).first());
                    }
                    editContainer.find(":input:visible:first").focus();
                    return;
                }
            }

            if (next) {
                that.current(next);
            }

            that.wrapper.focus();
            if ((!isEdited && !next) || next) {
                if (mode == "incell") {
                    that.editCell(that.current());
                } else {
                    that.editRow(that.current().parent());
                }
            }
        },

        _wrapper: function() {
            var that = this,
                table = that.table,
                height = that.options.height,
                wrapper = that.element;

            if (!wrapper.is("div")) {
               wrapper = wrapper.wrap("<div/>").parent();
            }

            that.wrapper = wrapper.addClass("k-grid k-widget")
                                  .attr(TABINDEX, math.max(table.attr(TABINDEX) || 0, 0));

            table.removeAttr(TABINDEX);

            if (height) {
                that.wrapper.css(HEIGHT, height);
                table.css(HEIGHT, "auto");
            }
        },

        _tbody: function() {
            var that = this,
                table = that.table,
                tbody;

            tbody = table.find(">tbody");

            if (!tbody.length) {
                tbody = $("<tbody/>").appendTo(table);
            }

            that.tbody = tbody;
        },

        _scrollable: function() {
            var that = this,
                header,
                table,
                options = that.options,
                height = that.wrapper.innerHeight(),
                scrollable = options.scrollable,
                scrollbar = kendo.support.scrollbar();

            if (scrollable) {
                header = that.wrapper.children().filter(".k-grid-header");

                if (!header[0]) {
                    header = $('<div class="k-grid-header" />').insertBefore(that.table);
                }

                // workaround for IE issue where scroll is not raised if container is same width as the scrollbar
                header.css("padding-right", scrollable.virtual ? scrollbar + 1 : scrollbar);
                table = $('<table cellspacing="0" />');
                table.append(that.thead);
                header.empty().append($('<div class="k-grid-header-wrap" />').append(table));

                that.content = that.table.parent();

                if (that.content.is(".k-virtual-scrollable-wrap")) {
                    that.content = that.content.parent();
                }

                if (!that.content.is(".k-grid-content, .k-virtual-scrollable-wrap")) {
                    that.content = that.table.wrap('<div class="k-grid-content" />').parent();

                    if (scrollable !== true && scrollable.virtual) {
                        new VirtualScrollable(that.content, {
                            dataSource: that.dataSource,
                            itemHeight: proxy(that._averageRowHeight, that)
                        });
                    }
                }

                var scrollables = header.find(">.k-grid-header-wrap, > .k-footer-template"); // add footer when implemented

                if (scrollable.virtual) {
                    that.content.find(">.k-virtual-scrollable-wrap").bind('scroll', function () {
                        scrollables.scrollLeft(this.scrollLeft);
                    });
                } else {
                    that.content.bind('scroll', function () {
                        scrollables.scrollLeft(this.scrollLeft);
                    });

                    kendo.touchScroller(that.content);
                }
            }
        },

        _setContentHeight: function() {
            var that = this,
                options = that.options,
                height = that.wrapper.innerHeight(),
                header = that.wrapper.children(".k-grid-header"),
                scrollbar = kendo.support.scrollbar();

            if (options.scrollable) {

                height -= header.outerHeight();

                if (that.pager) {
                    height -= that.pager.element.outerHeight();
                }

                if(options.groupable) {
                    height -= that.wrapper.children(".k-grouping-header").outerHeight();
                }

                if(options.toolbar) {
                    height -= that.wrapper.children(".k-grid-toolbar").outerHeight();
                }

                if (height > scrollbar * 2) { // do not set height if proper scrollbar cannot be displayed
                    that.content.height(height);
                }
            }
        },

        _averageRowHeight: function() {
            var that = this,
                rowHeight = that._rowHeight;

            if (!that._rowHeight) {
                that._rowHeight = rowHeight = that.table.outerHeight() / that.table[0].rows.length;
                that._sum = rowHeight;
                that._measures = 1;

                totalHeight = math.round(that.dataSource.total() * rowHeight);
            }

            var currentRowHeight = that.table.outerHeight() / that.table[0].rows.length;

            if (rowHeight !== currentRowHeight) {
                that._measures ++;
                that._sum += currentRowHeight;
                that._rowHeight = that._sum / that._measures;
            }
            return rowHeight;
        },

        _dataSource: function() {
            var that = this,
                options = that.options,
                pageable,
                dataSource = options.dataSource;

            dataSource = isArray(dataSource) ? { data: dataSource } : dataSource;

            if (isPlainObject(dataSource)) {
                extend(dataSource, { table: that.table, fields: that.columns });

                pageable = options.pageable;

                if (isPlainObject(pageable) && pageable.pageSize !== undefined) {
                    dataSource.pageSize = pageable.pageSize;
                }
            }

            if (that.dataSource && that._refreshHandler) {
                that.dataSource.unbind(CHANGE, that._refreshHandler)
                                .unbind(REQUESTSTART, that._requestStartHandler)
                                .unbind(ERROR, that._errorHandler);
            } else {
                that._refreshHandler = proxy(that.refresh, that);
                that._requestStartHandler = proxy(that._requestStart, that);
                that._errorHandler = proxy(that._error, that);
            }

            that.dataSource = DataSource.create(dataSource)
                                .bind(CHANGE, that._refreshHandler)
                                .bind(REQUESTSTART, that._requestStartHandler)
                                .bind(ERROR, that._errorHandler);
        },

        _error: function() {
            this._progress(false);
        },

        _requestStart: function() {
            this._progress(true);
        },

        _modelChange: function(e) {
            var that = this,
                model = e.model,
                row = that.tbody.find("tr[" + kendo.attr("uid") + "=" + model.uid +"]"),
                cell,
                column,
                isAlt = row.hasClass("k-alt"),
                tmp,
                idx,
                length;

            if (row.children(".k-edit-cell").length) {
                row.children(":not(.k-group-cell,.k-hierarchy-cell)").each(function() {
                    cell = $(this);
                    column = that.columns[that.cellIndex(cell)];

                    if (column.field === e.field) {
                        if (!cell.hasClass("k-edit-cell")) {
                            that._displayCell(cell, column, model);
                            $('<span class="k-dirty"/>').prependTo(cell);
                        } else {
                            cell.addClass("k-dirty-cell");
                        }
                    }
                });

            } else if (!row.hasClass("k-grid-edit-row")) {
                tmp = $((isAlt ? that.altRowTemplate : that.rowTemplate)(model));

                row.replaceWith(tmp);

                for (idx = 0, length = that.columns.length; idx < length; idx++) {
                    column = that.columns[idx];

                    if (column.field === e.field) {
                        cell = tmp.children(":not(.k-group-cell,.k-hierarchy-cell)").eq(idx)
                        $('<span class="k-dirty"/>').prependTo(cell);
                    }
                }
            }
        },

        _pageable: function() {
            var that = this,
                wrapper,
                pageable = that.options.pageable;

            if (pageable) {
                wrapper = that.wrapper.children("div.k-grid-pager").empty();

                if (!wrapper.length) {
                    wrapper = $('<div class="k-pager-wrap k-grid-pager"/>').appendTo(that.wrapper);
                }

                if (that.pager) {
                    that.pager.destroy();
                }

                if (typeof pageable === "object" && pageable instanceof kendo.ui.Pager) {
                    that.pager = pageable;
                } else {
                    that.pager = new kendo.ui.Pager(wrapper, extend({}, pageable, { dataSource: that.dataSource }));
                }
            }
        },

        _footer: function() {
            var that = this,
                aggregates = that.dataSource.aggregates(),
                html = "",
                footerTemplate = that.footerTemplate,
                options = that.options;

            if (footerTemplate) {
                html = $(that._wrapFooter(footerTemplate(aggregates || {})));

                if (that.footer) {
                    var tmp = html;

                    that.footer.replaceWith(tmp);
                    that.footer = tmp;
                } else {
                    if (options.scrollable) {
                        that.footer = options.pageable ? html.insertBefore(that.wrapper.children("div.k-grid-pager")) : html.appendTo(that.wrapper);
                    } else {
                        that.footer = html.insertBefore(that.tbody);
                    }
                }
            }
        },

        _wrapFooter: function(footerRow) {
            var that = this,
                html = "",
                columns = that.columns,
                idx,
                length,
                groups = that.dataSource.group().length,
                column;

            if (that.options.scrollable) {
                html = $('<div class="k-grid-footer"><table cellspacing="0"><tbody>' + footerRow + '</tbody></table></div>');
                that._appendCols(html.find("table"));
                return html;
            }

            return '<tfoot>' + footerRow + '</tfoot>';
        },

        _filterable: function() {
            var that = this,
                columns = that.columns,
                filterable = that.options.filterable;

            if (filterable) {
                that.thead
                    .find("th:not(.k-hierarchy-cell)")
                    .each(function(index) {
                        if (columns[index].filterable !== false) {
                            $(this).kendoFilterMenu(extend(true, {}, filterable, columns[index].filterable, { dataSource: that.dataSource }));
                        }
                    })
            }
        },

        _sortable: function() {
            var that = this,
                columns = that.columns,
                column,
                sortable = that.options.sortable;

            if (sortable) {
                that.thead
                    .find("th:not(.k-hierarchy-cell)")
                    .each(function(index) {
                        column = columns[index];
                        if (column.sortable !== false && !column.command) {
                            $(this).kendoSortable(extend({}, sortable, { dataSource: that.dataSource }));
                        }
                    })
            }
        },

        _columns: function(columns) {
            var that = this,
                table = that.table,
                encoded,
                cols = table.find("col"),
                dataSource = that.options.dataSource;

            // using HTML5 data attributes as a configuration option e.g. <th data-field="foo">Foo</foo>
            columns = columns.length ? columns : map(table.find("th"), function(th, idx) {
                var th = $(th),
                    sortable = th.attr(kendo.attr("sortable")),
                    filterable = th.attr(kendo.attr("filterable")),
                    type = th.attr(kendo.attr("type")),
                    groupable = th.attr(kendo.attr("groupable")),
                    field = th.attr(kendo.attr("field"));

                if (!field) {
                   field = th.text().replace(/\s|[^A-z0-9]/g, "");
                }

                return {
                    field: field,
                    type: type,
                    sortable: sortable !== "false",
                    filterable: filterable !== "false",
                    groupable: groupable !== "false",
                    template: th.attr(kendo.attr("template")),
                    width: cols.eq(idx).css("width")
                };
            });

            encoded = !(that.table.find("tbody tr").length > 0 && (!dataSource || !dataSource.transport));

            that.columns = map(columns, function(column) {
                column = typeof column === STRING ? { field: column } : column;
                return extend({ encoded: encoded }, column);
            });
        },

        _tmpl: function(rowTemplate, alt) {
            var that = this,
                settings = extend({}, kendo.Template, that.options.templateSettings),
                paramName = settings.paramName,
                idx,
                length = that.columns.length,
                template,
                state = { storage: {}, count: 0 },
                id,
                column,
                type,
                hasDetails = that._hasDetails(),
                className = [],
                groups = that.dataSource.group().length;

            if (!rowTemplate) {
                rowTemplate = "<tr";

                if (alt) {
                    className.push("k-alt");
                }

                if (hasDetails) {
                    className.push("k-master-row");
                }

                if (className.length) {
                    rowTemplate += ' class="' + className.join(" ") + '"';
                }

                if (length) { // data item is an object
                    rowTemplate += ' ' + kendo.attr("uid") + '="#=uid#"';
                }

                rowTemplate += ">";

                if (groups > 0) {
                    rowTemplate += groupCells(groups);
                }

                if (hasDetails) {
                    rowTemplate += '<td class="k-hierarchy-cell"><a class="k-icon k-plus" href="\\#"></a></td>';
                }

                for (idx = 0; idx < length; idx++) {
                    column = that.columns[idx];
                    template = column.template;
                    type = typeof template;

                    rowTemplate += "<td>";
                    rowTemplate += that._cellTmpl(column, state);

                    rowTemplate += "</td>";
                }

                rowTemplate += "</tr>";
            }

            rowTemplate = kendo.template(rowTemplate, settings);

            if (state.count > 0) {
                return proxy(rowTemplate, state.storage);
            }

            return rowTemplate;
        },

        _cellTmpl: function(column, state) {
            var that = this,
                settings = extend({}, kendo.Template, that.options.templateSettings),
                template = column.template,
                paramName = settings.paramName,
                html = "",
                idx,
                length,
                format = column.format,
                type = typeof template;

            if (column.command) {
                if (isArray(column.command)) {
                    for (idx = 0, length = column.command.length; idx < length; idx++) {
                        html += that._createButton(column.command[idx]);
                    }
                    return html.replace(templateHashRegExp, "\\#");
                }
                return that._createButton(column.command).replace(templateHashRegExp, "\\#");
            }

            if (type === FUNCTION) {
                state.storage["tmpl" + state.count] = template;
                html += "#=this.tmpl" + state.count + "(" + paramName + ")#";
                state.count ++;
            } else if (type === STRING) {
                html += template;
            } else {
                html += column.encoded ? "${" : "#=";

                if (format) {
                    html += 'kendo.format(\"' + format.replace(formatRegExp,"\\}") + '\",';
                }

                if (!settings.useWithBlock) {
                    html += paramName + ".";
                }

                html += column.field;

                if (format) {
                    html += ")";
                }

                html += column.encoded ? "}" : "#";
            }
            return html;
        },

        _templates: function() {
            var that = this,
                options = that.options,
                dataSource = that.dataSource,
                groups = dataSource.group(),
                aggregates = dataSource.aggregate();

            that.rowTemplate = that._tmpl(options.rowTemplate);
            that.altRowTemplate = that._tmpl(options.altRowTemplate || options.rowTemplate, true);

            if (that._hasDetails()) {
                that.detailTemplate = that._detailTmpl(options.detailTemplate || "");
            }

            if (!isEmptyObject(aggregates) ||
                $.grep(that.columns, function(column) { return column.footerTemplate }).length) {

                that.footerTemplate = that._footerTmpl(aggregates, "footerTemplate", "k-footer-template");
            }
            if (groups.length && $.grep(that.columns, function(column) { return column.groupFooterTemplate }).length) {
                aggregates = $.map(groups, function(g) { return g.aggregates });
                that.groupFooterTemplate = that._footerTmpl(aggregates, "groupFooterTemplate", "k-group-footer");
            }
        },

        _footerTmpl: function(aggregates, templateName, rowClass) {
            var that = this,
                settings = extend({}, kendo.Template, that.options.templateSettings),
                paramName = settings.paramName,
                html = "",
                idx,
                length,
                columns = that.columns,
                template,
                type,
                storage = {},
                count = 0,
                scope = {},
                dataSource = that.dataSource,
                groups = dataSource.group().length,
                fieldsMap = {},
                column;

            if (!isEmptyObject(aggregates)) {
                if (isArray(aggregates)) {
                    for (idx = 0, length = aggregates.length; idx < length; idx++) {
                        fieldsMap[aggregates[idx].field] = true;
                    }
                } else {
                    fieldsMap[aggregates.field] = true;
                }
            }

            html += '<tr class="' + rowClass + '">';

            if (groups > 0) {
                html += groupCells(groups);
            }

            if (that._hasDetails()) {
                html += '<td class="k-hierarchy-cell">&nbsp;</td>';
            }

            for (idx = 0, length = that.columns.length; idx < length; idx++) {
                column = columns[idx];
                template = column[templateName];
                type = typeof template;

                html += "<td>";

                if (template) {
                    if (type !== FUNCTION) {
                        scope = fieldsMap[column.field] ? extend({}, settings, { paramName: paramName + "." + column.field }) : {};
                        template = kendo.template(template, scope);
                    }

                    storage["tmpl" + count] = template;
                    html += "#=this.tmpl" + count + "(" + paramName + ")#";
                    count ++;
                } else {
                    html += "&nbsp;";
                }

                html += "</td>";
            }

            html += '</tr>';

            html = kendo.template(html, settings);

            if (count > 0) {
                return proxy(html, storage);
            }

            return html;
        },

        _detailTmpl: function(template) {
            var that = this,
                html = "",
                settings = extend({}, kendo.Template, that.options.templateSettings),
                paramName = settings.paramName,
                templateFunctionStorage = {},
                templateFunctionCount = 0,
                groups = that.dataSource.group().length,
                columns = that.columns.length,
                type = typeof template;

            html += '<tr class="k-detail-row">';
            if (groups > 0) {
                html += groupCells(groups);
            }
            html += '<td class="k-hierarchy-cell"></td><td class="k-detail-cell"' + (columns ? ' colspan="' + columns + '"' : '') + ">";

            if (type === FUNCTION) {
                templateFunctionStorage["tmpl" + templateFunctionCount] = template;
                html += "#=this.tmpl" + templateFunctionCount + "(" + paramName + ")#";
                templateFunctionCount ++;
            } else {
                html += template;
            }

            html += "</td></tr>";

            html = kendo.template(html, settings);

            if (templateFunctionCount > 0) {
                return proxy(html, templateFunctionStorage);
            }

            return html;
        },

        _hasDetails: function() {
            var that = this;

            return that.options.detailTemplate !== undefined  || (that._events[DETAILINIT] || []).length;
        },

        _details: function() {
            var that = this;

            that.table.delegate(".k-hierarchy-cell .k-plus, .k-hierarchy-cell .k-minus", CLICK, function(e) {
                var button = $(this),
                    expanding = button.hasClass("k-plus"),
                    masterRow = button.closest("tr.k-master-row"),
                    detailRow,
                    detailTemplate = that.detailTemplate,
                    data,
                    hasDetails = that._hasDetails();

                button.toggleClass("k-plus", !expanding)
                    .toggleClass("k-minus", expanding);

                if(hasDetails && !masterRow.next().hasClass("k-detail-row")) {
                    data = that.dataItem(masterRow),
                    $(detailTemplate(data)).insertAfter(masterRow);

                    that.trigger(DETAILINIT, { masterRow: masterRow, detailRow: masterRow.next(), data: data, detailCell: masterRow.next().find(".k-detail-cell") });
                }

                detailRow = masterRow.next();

                that.trigger(expanding ? DETAILEXPAND : DETAILCOLLAPSE, { masterRow: masterRow, detailRow: detailRow});
                detailRow.toggle(expanding);

                e.preventDefault();
                return false;
            });
        },

        /**
         * Returns the data item to which a given table row (tr DOM element) is bound.
         * @param {Selector | DOM Element} tr Target row.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // returns the data item for first row
         * grid.dataItem(grid.tbody.find(">tr:first"));
         */
        dataItem: function(tr) {
            return this._data[this.tbody.find('> tr:not(.k-grouping-row,.k-detail-row)').index($(tr))]
        },

        /**
         * Expands specified master row.
         * @param {Selector | DOM Element} row Target master row to expand.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // expands first master row
         * grid.expandRow(grid.tbody.find(">tr.k-master-row:first"));
         */
        expandRow: function(tr) {
            $(tr).find('> td .k-plus, > td .k-expand').click();
        },

        /**
         * Collapses specified master row.
         * @param {Selector | DOM Element} row Target master row to collapse.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // collapses first master row
         * grid.collapseRow(grid.tbody.find(">tr.k-master-row:first"));
         */
        collapseRow: function(tr) {
            $(tr).find('> td .k-minus, > td .k-collapse').click();
        },

        _thead: function() {
            var that = this,
                columns = that.columns,
                hasDetails = that._hasDetails() && columns.length,
                idx,
                length,
                html = "",
                thead = that.table.find("thead"),
                tr,
                th;

            if (!thead.length) {
                thead = $("<thead/>").insertBefore(that.tbody);
            }

            tr = that.table.find("tr").filter(":has(th)");

            if (!tr.length) {
                tr = thead.children().first();
                if(!tr.length) {
                    tr = $("<tr/>");
                }
            }

            if (!tr.children().length) {
                if (hasDetails) {
                    html += '<th class="k-hierarchy-cell">&nbsp;</th>';
                }

                for (idx = 0, length = columns.length; idx < length; idx++) {
                    th = columns[idx];

                    if (!th.command) {
                        html += "<th " + kendo.attr("field") + "='" + th.field + "' ";
                        if (th.title) {
                            html += kendo.attr("title") + '="' + th.title.replace(/'/g, "\'") + '" ';
                        }

                        if (th.groupable !== undefined) {
                            html += kendo.attr("groupable") + "='" + th.groupable + "' ";
                        }

                        if (th.aggregates) {
                            html += kendo.attr("aggregates") + "='" + th.aggregates + "'";
                        }

                        html += ">" + (th.title || th.field || "") + "</th>";
                    } else {
                        html += "<th>" + (th.title || "") + "</th>";
                    }
                }

                tr.html(html);
            } else if (hasDetails) {
                tr.prepend('<th class="k-hierarchy-cell">&nbsp;</th>');
            }

            tr.find("th").addClass("k-header");

            if(!that.options.scrollable) {
                thead.addClass("k-grid-header");
            }

            tr.appendTo(thead);

            that.thead = thead;

            that._sortable();

            that._filterable();

            that._scrollable();

            that._updateCols();

            that._setContentHeight();
        },

        _updateCols: function() {
            var that = this;

            that._appendCols(that.thead.parent().add(that.table));
        },

        _appendCols: function(table) {
            var that = this,
                colgroup = table.find("colgroup"),
                width,
                cols = map(that.columns, function(column) {
                    width = column.width;
                    if (width && parseInt(width) != 0) {
                        return kendo.format('<col style="width:{0}"/>', typeof width === STRING? width : width + "px");
                    }

                    return "<col />";
                }),
                groups = that.dataSource.group().length;

            if (that._hasDetails()) {
                cols.splice(0, 0, '<col class="k-hierarchy-col" />');
            }

            if (colgroup.length) {
                colgroup.remove();
            }

            colgroup = $("<colgroup/>").append($(new Array(groups + 1).join('<col class="k-group-col">') + cols.join("")));

            table.prepend(colgroup);
        },

        _autoColumns: function(schema) {
            if (schema && schema.toJSON) {
                var that = this,
                    field;

                schema = schema.toJSON();

                for (field in schema) {
                    that.columns.push({ field: field });
                }

                that._thead();

                that._templates();
            }
        },

        _rowsHtml: function(data) {
            var that = this,
                html = "",
                idx,
                length,
                rowTemplate = that.rowTemplate,
                altRowTemplate = that.altRowTemplate;

            for (idx = 0, length = data.length; idx < length; idx++) {
                if (idx % 2) {
                    html += altRowTemplate(data[idx]);
                } else {
                    html += rowTemplate(data[idx]);
                }

                that._data.push(data[idx]);
            }

            return html;
        },

        _groupRowHtml: function(group, colspan, level) {
            var that = this,
                html = "",
                idx,
                length,
                field = group.field,
                column = $.grep(that.columns, function(column) { return column.field == field; })[0] || { },
                value = column.format ? kendo.format(column.format, group.value) : group.value,
                template = column.groupHeaderTemplate,
                text =  (column.title || field) + ': ' + value,
                data = extend({}, { field: group.field, value: group.value }, group.aggregates[group.field]),
                groupItems = group.items;

            if (template) {
                text  = typeof template === FUNCTION ? template(data) : kendo.template(template)(data);
            }

            html +=  '<tr class="k-grouping-row">' + groupCells(level) +
                      '<td colspan="' + colspan + '">' +
                        '<p class="k-reset">' +
                         '<a class="k-icon k-collapse" href="#"></a>' + text
                          +'</p></td></tr>';

            if(group.hasSubgroups) {
                for(idx = 0, length = groupItems.length; idx < length; idx++) {
                    html += that._groupRowHtml(groupItems[idx], colspan - 1, level + 1);
                }
            } else {
                html += that._rowsHtml(groupItems);
            }

            if (that.groupFooterTemplate) {
                html += that.groupFooterTemplate(group.aggregates);
            }
            return html;
        },

        /**
         * Collapses specified group.
         * @param {Selector | DOM Element} group Target group item to collapse.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // collapses first group item
         * grid.collapseGroup(grid.tbody.find(">tr.k-grouping-row:first"));
         */
        collapseGroup: function(group) {
            group = $(group).find(".k-icon").addClass("k-expand").removeClass("k-collapse").end();
            var level = group.find(".k-group-cell").length;

            group.nextUntil(function() {
                return $(".k-group-cell", this).length <= level;
            }).hide();
        },

        /**
         * Expands specified group.
         * @param {Selector | DOM Element} group Target group item to expand.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // expands first group item
         * grid.expandGroup(grid.tbody.find(">tr.k-grouping-row:first"));
         */
        expandGroup: function(group) {
            group = $(group).find(".k-icon").addClass("k-collapse").removeClass("k-expand").end();
            var that = this,
                level = group.find(".k-group-cell").length;

            group.nextAll("tr").each(function () {
                var tr = $(this);
                var offset = tr.find(".k-group-cell").length;
                if (offset <= level)
                    return false;

                if (offset == level + 1) {
                    tr.show();

                    if (tr.hasClass("k-grouping-row") && tr.find(".k-icon").hasClass("k-collapse"))
                        that.expandGroup(tr);
                }
            });
        },

        _updateHeader: function(groups) {
            var that = this,
                cells = that.thead.find("th.k-group-cell"),
                length = cells.length;

            if(groups > length) {
                $(new Array(groups - length + 1).join('<th class="k-group-cell k-header">&nbsp;</th>')).prependTo(that.thead.find("tr"));
            } else if(groups < length) {
                length = length - groups;
                $($.grep(cells, function(item, index) { return length > index } )).remove();
            }
        },

        _firstDataItem: function(data, grouped) {
            if(data && grouped) {
                if(data.hasSubgroups) {
                    data = this._firstDataItem(data.items[0], grouped);
                } else {
                    data = data.items[0];
                }
            }
            return data;
        },

        _progress: function(toggle) {
            var that = this,
                element = that.element.is("table") ? that.element.parent() : (that.content && that.content.length ? that.content : that.element);

            kendo.ui.progress(element, toggle);
        },

        /**
         * Reloads the data and repaints the grid.
         * @example
         * // get a reference to the grid widget
         * var grid = $("#grid").data("kendoGrid");
         * // refreshes the grid
         * grid.refresh();
         */
        refresh: function(e) {
            var that = this,
                length,
                idx,
                html = "",
                data = that.dataSource.view(),
                tbody,
                placeholder,
                currentIndex,
                current = that.current(),
                groups = (that.dataSource.group() || []).length,
                colspan = groups + that.columns.length;

            if (e && e.action === "itemchange" && that.editable) { // skip rebinding if editing is in progress
                return;
            }

            that.trigger("dataBinding");

            if (current && current.hasClass("k-state-focused")) {
                currentIndex = that.items().index(current.parent());
            }

            that._distroyEditable();

            that._progress(false);

            that._data = [];

            if (!that.columns.length) {
                that._autoColumns(that._firstDataItem(data[0], groups));
                colspan = groups + that.columns.length;
            }

            that._group = groups > 0 || that._group;

            if(that._group) {
                that._templates();
                that._updateCols();
                that._updateHeader(groups);
                that._group = groups > 0;
            }

            if(groups > 0) {

                if (that.detailTemplate) {
                    colspan++;
                }

                for (idx = 0, length = data.length; idx < length; idx++) {
                    html += that._groupRowHtml(data[idx], colspan, 0);
                }
            } else {
                html += that._rowsHtml(data);
            }

            if (tbodySupportsInnerHtml) {
                that.tbody[0].innerHTML = html;
            } else {
                placeholder = document.createElement("div");
                placeholder.innerHTML = "<table><tbody>" + html + "</tbody></table>";
                tbody = placeholder.firstChild.firstChild;
                that.table[0].replaceChild(tbody, that.tbody[0]);
                that.tbody = $(tbody);
            }

            that._footer();

            if (currentIndex >= 0) {
                that.current(that.items().eq(currentIndex).children().filter(DATA_CELL).first());
            }

            that.trigger(DATABOUND);
       }
   });

   function hasCommand(commands, name) {
       var idx, length, command;

       if (typeof commands === STRING) {
           return commands === name;
       }

       if (isArray(commands)) {
           for (idx = 0, length = commands.length; idx < length; idx++) {
               command = commands[idx];

               if ((typeof command === STRING && command === name) || (command.name === name)) {
                   return true;
               }
           }
       }
       return false;
   };

   ui.plugin(Grid);
   ui.plugin(VirtualScrollable);
})(jQuery);
