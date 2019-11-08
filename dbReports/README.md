# Torrent Browser 

## Internalization Support - i18n

### Key Areas
* Dangjo Templates
* Client-Side JS and static files
* Django Server-Side 
    * Views, Models, APIS
    * tbd. 


### Configuration

#### Django Settings - iondb.settings

    ...
    
    LANGUAGES = (
    # English
    # ('en', u'English'), # instead of 'en'
    # English United Stated
    ('en-us', u'English US'), # instead of 'en_US'
    # Russian
    ('ru', u'русский'), # instead of 'ru'
    # Simplified Chinese
    ('zh-cn', u'简体中1'), # instead of 'zh-CN'
    #Traditional Chinese
    # ('zh-tw', u'繁體中文'), # instead of 'zh-TW'
    )
    ...
    # http://django-docs.jarvis.itw/ref/settings.html?highlight=locale_paths#std:setting-LOCALE_PATHS
    LOCALE_PATHS = (
        '/opt/ion/iondb/locale',
        '/var/local/translations/locale' # placeholder for additional translations, e.g. from oem provided .deb
    )
    # If you set this to False, Django will make some optimizations so as not
    # to load the internationalization machinery.
    USE_I18N = True
    ...
    MIDDLEWARE_CLASSES = (
        'iondb.rundb.middleware.ChangeRequestMethodMiddleware',
        'django.middleware.common.CommonMiddleware',
        'iondb.rundb.middleware.DeleteSessionOnLogoutMiddleware',
        'django.contrib.sessions.middleware.SessionMiddleware',
        ...
        # 'django.middleware.locale.LocaleMiddleware',  #uncomment to allow end-user locale to dictate the language used
        'iondb.bin.startup_housekeeping.StartupHousekeeping'
    )

tbd..