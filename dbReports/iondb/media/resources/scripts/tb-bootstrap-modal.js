(function($) {    
    if ($.fn.modalmanager) {
        $.fn.modal.defaults = {
            keyboard: false, 
            backdrop: 'static',
            loading: false,
            show: true,
            width: null,
            height: null,
            maxHeight: null,
            modalOverflow: false,  // set on a modal by using data-modal-Overflow="true"
            consumeTab: true,
            focusOn: null,
            attentionAnimation: '', //'shake' or ''
            manager: 'body',
            spinner: '<div class="loading-spinner" style="width: 200px; margin-left: -100px;"><div class="progress progress-striped active"><div class="bar" style="width: 100%;"></div></div></div>'
        };
        $('body').modalmanager('loading');
    } else {
        $.fn.modal.defaults = {
              backdrop: 'static'
            , keyboard: false
            , show: true
        }
    }
})(jQuery); 