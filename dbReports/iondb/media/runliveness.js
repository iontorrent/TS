var _TRIES = 0;
var MAX_TRIES = 3;
var _FOUND_STATUS = false;
var _LIVENESS = false;	

function analysis_liveness(url, domid) {
    function cb(data) {
        if (_FOUND_STATUS) {
            return;
        }
        if (data["success"] || data["status"] == "Failed") {
            _FOUND_STATUS = true;
            var jqele = $("#" + domid + " > ul");
            var ele = jqele.get(0);
            var report = $("<a />");
            var log = $("<a />");
            var anchors = [report, log];
            var hrefs = [data["report"], data["log"]];
            var text = ["Report", "Log"];
            if (data["exists"]) {
                for (var i in anchors) {
                    var a = anchors[i];
                    var li = $("<li />");
                    a.attr("href", hrefs[i]);
                    a.text(text[i]);
                    li.get(0).appendChild(a.get(0));
                    ele.appendChild(li.get(0));
                }
                var sp = $("#" + domid + " > span");
                var timeout_cb = function () {

                    sp.hide("fast", function () {
                        sp.text("Status: " + data["status"]);
                        sp.show("fast", function () {
                            $("#links").show();
                        });
                    });
                };

                setTimeout(timeout_cb, 300.0);
            } else {
                $("#" + domid).html("Job is waiting to be processed, no reports have been generated yet. View the status of the job on the <a href='../reports'>Reports page</a>").effect("bounce");

            }


        } else if (data["status"] == "Unknown" && _TRIES < MAX_TRIES) {
            _TRIES++;
            var timeout_cb = function () {
                analysis_liveness(url, domid);
            };
            setTimeout(timeout_cb, 1000.0);
        } else {
            var msg = $("<div />");
            msg.text("The analysis could not be started.");
            $("#" + domid).get(0).appendChild(msg.get(0));
        }
    }

    //wait a few seconds before asking the server if the job started.
    _LIVENESS = setTimeout(function () {
        $.getJSON(url, null, cb);
    }, 3500);

}

function analysis_live_ready_cb() {
	_TRIES = 0;
	_FOUND_STATUS = false;
	_LIVENESS = true;
    var domid = "analysis_liveness_display";
    var a = $("#analysis_liveness_info > a");
    var url = a.attr("href");
    analysis_liveness(url, domid);
}

function analysis_liveness_off() {
	clearTimeout(_LIVENESS);
}

$(document).ready(analysis_live_ready_cb);