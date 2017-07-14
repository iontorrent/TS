function enableIframeResizing(iframe) {
    iframe.iframeAutoHeight({
        triggerFunctions: [function (resizeFunction, iframe) {
            //Always try to resize the iframe
            setInterval(function () {
                resizeFunction(iframe);
            }, 500);
        }],
        heightCalculationOverrides: ['webkit', 'mozilla', 'msie', 'opera', 'chrome'].map(function (browser) {
            return {
                browser: browser,
                calculation: function (iframe, $iframeBody, options) {
                    return $(iframe).contents().find("html").height() + options.heightOffset;
                }
            }
        })
    });
}