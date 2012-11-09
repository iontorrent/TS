function getGuruBaseUrl() {
    return 'http://torrentguru.iontorrent.com';
}

function getPluginConfigFromDb(successMethod){
    $.ajax({
        url:"/rundb/api/v1/plugin/" + TB_plugin.pk + "/?format=json",
        dataType:"json",
        type: "GET",
        async: false,
        success:function(data){
            successMethod(data);
        }
    });
}

function validateTsInfo(tsUrl, tsUsername, tsPassword, successMethod, failMethod){
    $.ajax({
        url: tsUrl + "/rundb/api/v1/experiment/?format=json&limit=1",
        dataType:"json",
        type: "GET",
        async: false,
        beforeSend: function (req) {
            req.setRequestHeader('Authorization', tsUsername + ":" + tsPassword);
        },
        success: function(data){
            successMethod(data);
        },
        error: function(data){
            failMethod(data);
        }
    });
}

function getDefaultTsInfo(successMethod){
    $.ajax({
        url:"/rundb/api/v1/globalconfig/?format=json&limit=1",
        dataType:"json",
        type: "GET",
        async: false,
        success:function(data){
            var retval = {}
            if (data.web_root != null && data.web_root.length > 0) {
                retval["ts_url"] = data.web_root
            } else {
                retval["ts_url"] = "http://localhost"
            }
            retval["ts_username"] = "ionuser";
            retval["ts_password"] = "ionuser";
            successMethod(retval);
        }
    });
}
