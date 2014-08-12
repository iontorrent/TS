# Copyright (C) 2010 Ion Torrent Systems, Inc. All Rights Reserved
# -*- coding: utf-8 -*- 
from django.shortcuts import render_to_response
import requests
import feedparser
import dateutil
from django.conf import settings
from iondb.rundb.models import NewsPost, GlobalConfig
from django.template import RequestContext
from django.utils import timezone
from django.contrib.auth.decorators import login_required


@login_required
def news(request):
	profile = request.user.userprofile
	ctx = {
		"articles": list(NewsPost.objects.all().order_by('-updated')),
		"last_read": profile.last_read_news_post,
		"is_updating": GlobalConfig.get().check_news_posts,

	}
	profile.last_read_news_post = timezone.now()
	profile.save()
	return render_to_response("rundb/extra/news.html", ctx, context_instance=RequestContext(request))
