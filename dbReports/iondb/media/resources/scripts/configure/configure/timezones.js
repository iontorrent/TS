$(function() {//DOM ready handler
    $('#auto_detect').click(function (e) {

        function change_timezone(data) {
            function get_browser_timezone_country_city() {
                var region1 = new Intl.DateTimeFormat('en-us');
                var options1 = region1.resolvedOptions();
                var result =  {
                    'current_zone': options1.timeZone.split('/')[0]
                    , 'current_city': options1.timeZone.split('/')[1]
                }
                $("#timezone_saved").html('');
                $("#timezone_saved").append( "<h4>" + $("#timezone_saved").data('msgAutoDetectTimezoneSuccess') + "</h4>");//Auto detect is complete.
                return result;
            }

            data = data || get_browser_timezone_country_city()
            $('#zone_select').val(data['current_zone']);
            $('#zone_select').trigger('change');
            //wait until the change is done, and then change the city!
            $('#city_select').val(data['current_city']);
        }

        $("#auto_detect").html($("#auto_detect").data('valueInprogress')); //"Detecting..."
        $("#timezone_saved").empty();
        e.preventDefault();
        e.stopPropagation();
        url = $(this).attr('href');
        formdata= "auto_detect"
        console.log(url, formdata)
        $.ajax({
            url : url,
            type : "POST",
        })
        .done(function(data){
            console.log(data['current_zone'],data['current_city']);
            change_timezone(data);
            $("#auto_detect").html($("#auto_detect").data('value'));
            $("#timezone_saved").append( "<h4>" + $("#timezone_saved").data('msgAutoDetectTimezoneSuccess') + "</h4>");//Auto detect is complete.
        })
        .fail(function(data){
            console.log("error")
            $("#timezone_saved").append( "<h4>" + $("#timezone_saved").data('msgAutoDetectTimezoneFailed') + "</h4>");//Auto Detect failed.
            change_timezone();
            $("#auto_detect").html($("#auto_detect").data('value'));
        });

    });

    $("#zone_select").change(function() {
        url = "get_all_cities/" + $('#zone_select').val() +'/'
        console.log(url)

        $.ajax({
            type: "GET",
            async: false,
            url: url
        })
        .done(function(data){
            $("#city_select").empty();
            $.each(data, function(key,val){
                for( values in val){
                    if( $(this).value == val[values] ){
                        $("#city_select").append( "<option value=" + val[values] + "selected>" + val[values] + "</option>");
                    }
                    else{
                        $("#city_select").append( "<option value=" + val[values] + ">" + val[values] + "</option>");
                    }
                }
                    
            });
         });
    });

	$('#timezone').submit(function(e) {
        $("#submit_button").val($("#submit_button").data('valueInprogress'));//"Submitting"
        //prevents default form submission behavior
        e.preventDefault();
        $("#timezone_saved").empty();

        var formdata = $('#timezone').serialize();
        var url = $('#timezone').attr('action'), type = $('#timezone').attr('method');

        console.debug(url, type, formdata);

        $.ajax({
            url : url,
            type : type,
            dataType : "html",
            data : formdata
        })
        .done(function(data){
           $("#timezone_saved").append( "<h4>" + $("#timezone_saved").data('msgSuccess') + "</h4>");//"Your timezone was set successfully"
           $("#submit_button").val($("#submit_button").data('value'));
        })
        .fail(function(data){
            $("#timezone_saved").append( "<h4>" + $("#timezone_saved").data('msgFailed') + "</h4>");//"Timezone update failed"
            $("#submit_button").val($("#submit_button").data('value'));
        });

    });
});
    