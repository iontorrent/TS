# Copyright (C) 2013 Ion Torrent Systems, Inc. All Rights Reserved

try:
    from django.conf.urls import patterns, url
except ImportError:
    # Compat Django 1.4
    from django.conf.urls.defaults import patterns, url

urlpatterns = patterns(
    'iondb.rundb.sample',
    url(r'^$', 'views.show_samplesets', name="samplesets"),
    
    url(r'^sampleset/import/$', 'views.show_import_samplesetitems', name="import_samples"),     
    url(r'^sampleset/import/save/$', 'views.save_import_samplesetitems', name="save_import_samples"),   
    url(r'^sampleset/import/download/$', 'views.download_samplefile_format', name="download_samplefile_format"),
        
    url(r'^sampleset/add/$', 'views.show_edit_sampleset', name='add_sampleset'),
    url(r'^sampleset/(\d+)/edit/$', 'views.show_edit_sampleset', name='edit_sampleset'),
    url(r'^sampleset/edited/$', 'views.save_sampleset', name='save_sampleset'),   
    url(r'^sampleset/added/$', 'views.save_sampleset', name='save_sampleset'), 
    url(r'^sampleset/(\d+)/delete/$', 'views.delete_sampleset', name="delete_sampleset"),      
    url(r'^sampleset/(\d+)/plan_run/$', 'views.show_plan_run', name='sampleset_plan_run'),

    url(r'^sampleattribute/$', 'views.show_sample_attributes', name="sample_attributes"),
        
    url(r'^sampleattribute/add/$', 'views.show_add_sample_attribute', name="add_sample_attribute"), 
    url(r'^sampleattribute/(\d+)/edit/$', 'views.show_edit_sample_attribute', name="edit_sample_attribute"),         
    url(r'^sampleattribute/save/$', 'views.save_sample_attribute', name="save_sample_attribute"),   
    
    url(r'^sampleattribute/(\d+)/toggle/$', 'views.toggle_visibility_sample_attribute', name="toggle_sample_attribute"),  
    url(r'^sampleattribute/(\d+)/delete/$', 'views.delete_sample_attribute', name="delete_sample_attribute"),  
    
    url(r'^samplesetitem/edited/$', 'views.save_samplesetitem', name='save_samplesetitem'),   
    url(r'^samplesetitem/added/$', 'views.save_samplesetitem', name='save_samplesetitem'), 
    url(r'^samplesetitem/(\d+)/edit/$', 'views.show_edit_sample_for_sampleset', name='edit_samplesetitem'), 
    url(r'^samplesetitem/(\d+)/remove/$', 'views.remove_samplesetitem', name='remove_samplesetitem'), 
         
    url(r'^samplesetitem/input/$', 'views.show_input_samplesetitems', name="input_samples"),
    url(r'^samplesetitem/input/getdata/$', 'views.get_input_samples_data', name="input_samples_data"),

    url(r'^samplesetitem/input/add/$', 'views.show_add_pending_samplesetitem', name="add_pending_sample"),   
    url(r'^samplesetitem/input/(\d+)/edit/$', 'views.show_edit_pending_samplesetitem', name="edit_pending_sample"),
    url(r'^samplesetitem/input/(\d+)/remove/$', 'views.remove_pending_samplesetitem', name="remove_pending_sample"),
    url(r'^samplesetitem/input/edited_pending/$', 'views.save_samplesetitem', name='update_edited_pending_sample'),   

    url(r'^samplesetitem/input_save/$', 'views.show_save_input_samples_for_sampleset', name="show_save_input_samples"),
    url(r'^samplesetitem/input/save/$', 'views.save_input_samples_for_sampleset', name="save_input_samples"),  
    url(r'^samplesetitem/input/clear/$', 'views.clear_input_samples_for_sampleset', name="clear_input_samples"),

    url(r'^libraryprepsummary/(?P<pk>\d+)/$', 'views.library_prep_summary', name='library_prep_summary'),
)
