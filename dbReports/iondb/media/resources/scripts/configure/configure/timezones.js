$(function() {//DOM ready handler
    $('#auto_detect').click(function (e) {
        $("#auto_detect").html("Detecting...");
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
            console.log(data['current_zone'],data['current_city'])
            $('#zone_select').val(data['current_zone']);
            $('#zone_select').trigger('change');
            //wait until the change is done, and then change the city!
            $('#city_select').val(data['current_city']);
            $("#auto_detect").html("Auto Detect Timezone");
            $("#timezone_saved").append( "<h4>"+ "Auto detect is complete." +"</h4>");
        })
        .fail(function(data){
            console.log("error")
            $("#timezone_saved").append( "<h4>"+ "ERROR: Auto Detect failed." +"</h4>");
            $("#auto_detect").html("Auto Detect Timezone");
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
        $("#submit_button").val("Submitting");
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
           $("#timezone_saved").append( "<h4>"+ "Your timezone was set successfully" +"</h4>");
           $("#submit_button").val("Save Time Zone");
        })
        .fail(function(data){
            $("#timezone_saved").append( "<h4>"+ "ERROR: Timezone update failed" +"</h4>");
            $("#submit_button").val("Save Time Zone");
        });

    });
});
    