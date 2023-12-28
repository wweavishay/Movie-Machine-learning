from django.contrib import admin
from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name = 'home'),
    path('movie/', views.movie, name = 'movie'),
    path('director/', views.director, name = 'director'),
    path('actor/', views.actor, name = 'actor'),
    path('prediction/', views.prediction, name = 'prediction'),
    path('recommendation/', views.recommendation, name = 'recommendation'),
    path('review/', views.review, name = 'review'),
    path('solution/', views.solution, name='solution'),
    path('insert_data/', views.insert_data, name = 'insert_data'),
    path('insert_data_submission/', views.insert_data_submission, name = 'insert_data_submission'),
    path('new_movie/', views.new_movie, name='new_movie'),
    path(r'^edit_movie/(?P<pk>\d+)/$', views.edit_movie, name='edit_movie'),
    path(r'^delete_movie/(?P<pk>\d+)/$', views.delete_movie, name='delete_movie'),
    path(r'^comment_movie/(?P<pk>\d+)/$', views.comment_movie, name='comment_movie'),
    path(r'^prediction/$', views.prediction, name='predict_search'),
    # url(r'^movie_detail/(?P<id>\d+)/$', views.detail, name='movie_detail'),
    path(r'^recommedation/$', views.search, name='search'),
]