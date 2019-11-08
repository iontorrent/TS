$(function() {
  uploader = new plupload.Uploader({
    container : 'bedPublish',
    max_file_size : '1000mb',
    runtimes: 'html5,flash,silverlight',
    browse_button: 'pickfiles',
    url: '/rundb/publish/plupload/BED/',
    chunk_size: '10mb',
    unique_names: false,
    multi_selection: false,
    multiple_queues: false,
    multipart_params: {meta: '{}'},
    silverlight_xap_url: '{% static "resources/plupload/js/Moxie.xap"%}',
    flash_swf_url: '{% static "resources/plupload/js/Moxie.swf"%}'
  });

  uploader.bind('Init', function(up, params) {
    // Don't do anyting special on init.
  });

  $('#upload_button').click(function() {
    uploader.settings.multipart_params.meta = JSON.stringify({
      hotspot: false,
      reference: null,
      upload_type: "ampliseq",
      choice: $('input:radio[name=instrument_choice]:checked').val()
    });
    uploader.start();
    return false;
  });

  uploader.init();

  uploader.bind('FilesAdded', function(up, files) {
    var file = files[0];
    if (up.files.length > 1) {
      up.removeFile(up.files[0]);
    }
    console.log(up.files);
    $('#file_name_info').html(file.name + ' (' + plupload.formatSize(file.size) + ')');
    $('#file_info').html('').hide();
    $('#file_progress').find('.bar').css('width', "0%");
    up.refresh(); // Reposition Flash/Silverlight
  });

  uploader.bind('UploadProgress', function(up, file) {
    $('#file_progress').show().find('.bar').css('width', file.percent + "%");
  });

  uploader.bind('Error', function(up, err) {
    $('#file_info').html($('#file_info').data('error')).show();
  });

  uploader.bind('FileUploaded', function(up, file) {
    $('#file_progress').find('.bar').css('width', file.percent + "%")
        .delay(500)
        .css('width', 0 + "%").parents('#file_progress').hide();
    $("#file_info").html($('#file_info').data('fileUploaded')).show();
    uploader.removeFile(file);
    $('#file_name_info').html($('#file_name_info').attr('placeholder'));
  });
});
