# user
USER_ID = 'user_id'
USER_ID_TYPE_CD = 'user_id_type_cd'
ITEM_ID = 'item_id'
LABEL = 'label'
ARTIST_ID_SET = 'artist_cd_set'
ARTIST_ID_SEP = '#'
LANGUAGE = 'language'
TAG_ID_SET = 'tag_id_set'
SONIC_ID_SET_NEW = 'sonic_id_set_new'
LANGUAGE_ID_SET_NEW = 'language_id_set_new'
ERAS_ID_SET_NEW = 'eras_id_set_new'
GENRE_ID_SET = 'genre_id_set'
SONIC_ID_SET = 'sonic_id_set'
ISVIP_UP = 'isvip_up'
AGE = 'age'
GENDER = 'gender'
PROVINCE = 'province'
SERIES_NAME = 'series_name'
COLOR = 'color'
UPDATE_TIME = 'update_time'
PLAY_SONG_COMPLETE_SET_3DY = 'play_song_complete_set_3dy'
SWITCH_SONG_SET_3DY = 'switch_song_set_3dy'
LAST_COLLECT_SONG_SET_3DY = 'last_collect_song_set_3dy'
LAST_COLLECT_SONG_SET_7DY = 'last_collect_song_set_7dy'
LAST_COLLECT_ARTIST_SET_7DY = 'last_collect_artist_set_7dy'
COLLECT_SONG_SET_1Y = 'collect_song_set_1y'
COLLECT_ARTIST_SET_1Y = 'collect_artist_set_1y'
LAST_DOWNLOAD_SONG_SET_7DY = 'last_download_song_set_7dy'
LAST_DISLIKE_SONG_SET_30DY = 'last_dislike_song_set_30dy'
LAST_DISLIKE_ARTIST_SET_30DY = 'last_dislike_artist_set_30dy'
USER_SONG_SET_SEP = '#'
USER_COLUMNS = [USER_ID, USER_ID_TYPE_CD, ITEM_ID, LABEL, ARTIST_ID_SET, LANGUAGE, TAG_ID_SET,
                SONIC_ID_SET_NEW, LANGUAGE_ID_SET_NEW, ERAS_ID_SET_NEW, GENRE_ID_SET, SONIC_ID_SET,
                ISVIP_UP, AGE, GENDER, PROVINCE, SERIES_NAME, COLOR, PLAY_SONG_COMPLETE_SET_3DY,
                SWITCH_SONG_SET_3DY, LAST_COLLECT_SONG_SET_3DY, LAST_COLLECT_SONG_SET_7DY,
                LAST_COLLECT_ARTIST_SET_7DY, COLLECT_SONG_SET_1Y, COLLECT_ARTIST_SET_1Y,
                LAST_DOWNLOAD_SONG_SET_7DY, LAST_DISLIKE_SONG_SET_30DY, LAST_DISLIKE_ARTIST_SET_30DY]

FULL_USER_COLUMNS = [USER_ID, AGE, GENDER, PROVINCE, PLAY_SONG_COMPLETE_SET_3DY, SWITCH_SONG_SET_3DY,
                     LAST_COLLECT_SONG_SET_7DY, LAST_COLLECT_ARTIST_SET_7DY, LAST_DOWNLOAD_SONG_SET_7DY,
                     LAST_DISLIKE_SONG_SET_30DY, LAST_DISLIKE_ARTIST_SET_30DY, UPDATE_TIME]
FULL_NECESSARY_USER_COLUMNS = [USER_ID, AGE, GENDER, PROVINCE, PLAY_SONG_COMPLETE_SET_3DY,
                               LAST_COLLECT_SONG_SET_7DY, LAST_DOWNLOAD_SONG_SET_7DY]
FULL_USER_COLUMN_SEP = '\u0001'

# song
SONG_ID = 'song_id'
SONG_NAME = 'song_name'
SONG_ARTIST_ID_SET = 'artist_cd_set'
SONG_ARTIST_NAME_SET = 'artist_name_set'
SONG_ARTIST_SEP = '/'
SONG_GROUP = 'song_group'
SONG_RECOMMEND_FLG = 'recommend_flg'
SONG_RELEASE_TIME = 'release_time'
SONG_EFF_TIME = 'eff_time'
SONG_CATEGORY_ID_SET = 'category_id_set'
SONG_CATEGORY_SEP = '/'
SONG_PLAY_CNT = 'play_cnt'
SONG_UPDATE_TIME = 'update_time'
SONG_TYPE = 'song_type'
ALBUM_ID = 'album_id'
ALBUM_NAME = 'album_name'
SONG_STATUS_CD = 'song_status_cd'
SONG_TYPE_CD = 'song_type_cd'
REV_LISTEN_CNT = 'rev_listen_cnt'
SONG_AUTHOR = 'song_author'
SONG_COMPOSER = 'song_composer'
SONG_NAME_TITLE = 'song_name_title'
SONG_CP = 'song_cp'
SONG_GENRES_TAG_ID = 'genres_tag_id'
SONG_GENRES_TAG_NAME = 'genres_tag_name'
SONG_ATTR_TAG_ID = 'attr_tag_id'
SONG_ATTR_TAG_NAME = 'attr_tag_name'
SONG_SCENE_TAG_ID = 'scene_tag_id'
SONG_SCENE_TAG_NAME = 'scene_tag_name'
SONG_THEME_TAG_ID = 'theme_tag_id'
SONG_THEME_TAG_NAME = 'theme_tag_name'
SONG_MOOD_TAG_ID = 'mood_tag_id'
SONG_MOOD_TAG_NAME = 'mood_tag_name'
SONG_LANGUA_TAG_ID = 'langua_tag_id'
SONG_LANGUA_TAG_NAME = 'langua_tag_name'
SONG_ERAS_TAG_ID = 'eras_tag_id'
SONG_ERAS_TAG_NAME = 'eras_tag_name'
SONG_TAG_SEP = '#'
ITEM_COLUMNS = [SONG_ID, SONG_NAME, SONG_ARTIST_ID_SET, SONG_ARTIST_NAME_SET, SONG_GROUP,
                SONG_RECOMMEND_FLG, SONG_RELEASE_TIME, SONG_EFF_TIME, SONG_CATEGORY_ID_SET,
                SONG_PLAY_CNT, SONG_UPDATE_TIME, SONG_TYPE]
FULL_ITEM_COLUMNS = [SONG_ID, SONG_NAME, ALBUM_ID, ALBUM_NAME, SONG_STATUS_CD, SONG_RECOMMEND_FLG,
                     SONG_TYPE_CD, SONG_EFF_TIME, REV_LISTEN_CNT, SONG_ARTIST_ID_SET,
                     SONG_ARTIST_NAME_SET, SONG_CATEGORY_ID_SET, SONG_GROUP, SONG_AUTHOR,
                     SONG_COMPOSER, SONG_NAME_TITLE, SONG_CP, SONG_GENRES_TAG_ID, SONG_GENRES_TAG_NAME,
                     SONG_ATTR_TAG_ID, SONG_ATTR_TAG_NAME, SONG_SCENE_TAG_ID, SONG_SCENE_TAG_NAME,
                     SONG_THEME_TAG_ID, SONG_THEME_TAG_NAME, SONG_MOOD_TAG_ID, SONG_MOOD_TAG_NAME,
                     SONG_LANGUA_TAG_ID, SONG_LANGUA_TAG_NAME, SONG_ERAS_TAG_ID, SONG_ERAS_TAG_NAME]

FULL_NECESSARY_ITEM_COLUMNS = [
    SONG_ID, SONG_NAME, SONG_ARTIST_NAME_SET, ALBUM_NAME, SONG_LANGUA_TAG_NAME, SONG_GENRES_TAG_NAME]
FULL_ITEM_COLUMN_SEP = '|'

GENDER_MAPPING = {
    'g_f': '女性',
    'g_m': '男性',
    '\\N': '性别未知的',
    '': '性别未知的'
}

AGE_MAPPING = {
    '1': '18岁以下',
    '2': '18-23岁',
    '3': '24-34岁',
    '4': '35-44岁',
    '5': '45-55岁',
    '6': '55岁以上',
    '\\N': '年龄未知',
    '': '年龄未知'
}

# lyric
LYRIC_CONTENT_ENCODE = 'content_encode'
LYRIC_CONTENT_TYPE = 'content_type'
LYRICS_URL = 'lyrics_url'
LINEBYLINE_LYRICS_URL = 'linebyline_lyrics_url'
WORDBYWORD_LYRICS_URL = 'wordbyword_lyrics_url'
LYRICS_INFO = 'lyrics_info'
LYRICS_AUDIT_LOG = 'audit_log'
LYRICS_ETL_TIME = 'etl_time'
LYRICS_CONTENT = 'lyrics_content'
LYRIC_PT_D = 'pt_d'

LYRIC_COLUMNS = [LYRIC_CONTENT_ENCODE, LYRIC_CONTENT_TYPE, LYRICS_URL, LINEBYLINE_LYRICS_URL, WORDBYWORD_LYRICS_URL,
                 LYRICS_INFO, LYRICS_AUDIT_LOG, LYRICS_ETL_TIME, LYRICS_CONTENT, LYRIC_PT_D]
