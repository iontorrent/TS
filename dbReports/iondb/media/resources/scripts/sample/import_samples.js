
$(function() {
	
	$('.modal_addSampleSet').click(function(e) {
		e.preventDefault();
        $(".add_sample_set_info").addClass("in");
		$(".add_sample_set_info").slideDown("fast");
	});

    //sample set filtering
    $("#sampleset_search_text").on("keyup", function(e) {
      //   $("input[name='sampleset']").parent().show();
      //   if ($(this).val()) {
    		// console.log("enter import_sampls.js - sampleset_search_text.change() value=", $(this).val());
    		
      //       $("input[name=sampleset]").parent().not(":contains(" + $(this).val() + ")").hide();
      //   }
      var term = $(this).val();
      var $select = $("#sampleset");
      $select.empty();
      var match = false;
      $.each(sampleset_items, function(k, v){
        if (term.length > 0) {
            var item = v.split(" ")[0];
            if (item.match(term)) {
                match = true;
                $select.append($("<option></option>", {"value" : k, "text" : v}));
            }
        } else {
            $select.append($("<option></option>", {"value" : k, "text" : v}));
        }
      });

    });

    //input file selected
    $("#postedfile").change(function(e) {
		console.log("enter import_sampls.js - postedfile.change() value=", $(this).val());
		$('.summaryfile').removeClass('hide')
		$('#summary_selectedFileName').text($(this).val()); 
    });
});


TB.namespace('TB.sample.batchupload');

TB.sample.batchupload.ready = function(sampleUrl) {

//    $('#modal_batch_planning_upload').on('hidden', function() {
//        $('body #modal_batch_planning_upload').remove();
//    });

    $(function() {
        jQuery.fn.uniform.language.required = gettext('uni-form-validation.language.required');
        $('#import_sample_upload').uniform({
            holder_class : 'control-group',
            msg_selector : 'p.help-block.error',
            error_class : 'alert alert-error',
            prevent_submit : false
        });

        $(".submitUpload").click(function(e) {        	
            e.preventDefault();
            $('#import_sample_upload').submit();
        });

        $.ajaxSetup({
            async : false
        });

        $('#import_sample_upload').ajaxForm({
            beforeSubmit : verify,
            success : handleResponse,
            error : AjaxError,
            dataType : 'json'
        });
    });

    //Check if there is a file
    function verify() {
    	console.log("at verify()...");
    	
        $("#postedfile").blur();

        var inputVal = $("#postedfile").val();
        if (!jQuery.fn.uniform.isValid($('#import_sample_upload'), jQuery.fn.uniform.defaults)) {
            return false;
        }

        //$('#import_sample_upload .modal-body #modal-error-messages').addClass('hide').empty();
    	//$('.main .container-fluid .content #modal-error-messages').addClass('hide').empty();
		$('.main .container-fluid .content .row-fluid .span8 #import_sample_upload #modal-error-messages').addClass('hide').empty();
		$('.main .container-fluid .content .row-fluid .span8 #import_sample_upload #modal-success-messages').addClass('hide').empty();

        //$("#loadingstatus").html("<div class='alert alert-info'><img style='height:30px;width:30px;' src='/site_media/resources/bootstrap/img/loading.gif'> Uploading csv file for plans </div>");
    }

    function AjaxError() {
        var _html = kendo.template($('#ImportSamplesFailureTemplate').html())({});
        $('.main .container-fluid .content .row-fluid .span8 #import_sample_upload #modal-error-messages')
            .removeClass('hide')
            .html(_html);
    }

    //handleResponse will handle both successful upload and validation errors
    function handleResponse(responseText, statusText) {
        console.log("responseText..", responseText);
        var _html = kendo.template($('#ImportSamplesResponseTemplate').html())(responseText);
        console.debug(_html);
        if (responseText.error) {
            $('.main .container-fluid .content .row-fluid .span8 #import_sample_upload #modal-error-messages')
                .removeClass('hide')
                .html(_html);
        }
        else  {
            $('.main .container-fluid .content .row-fluid .span8 #import_sample_upload #modal-success-messages')
                .removeClass('hide')
                .html(_html);
            window.location = sampleUrl;
        }
    }

};
