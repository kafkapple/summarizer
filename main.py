from tqdm import tqdm 
import os
import pandas as pd 
from config import Config
from fetch import YouTube, WebContent, RaindropClient, PocketClient#, DiffbotExtractor
from summarizer import BaseSummarizer#, TextSummarizer, VideoSummarizer
from logger import Pocket2Notion, YouTube2Notion, Raindrop2Notion
from utils import Utils
import logging
logging.basicConfig(level=logging.INFO)
utils = Utils()

@utils.timeit
def summarize_youtube(config, summarizer, log_youtube):
    youtube = YouTube(config)
    playlist_url = 'https://youtube.com/playlist?list=PLuLudIpu5Viluk0AXnWT1nC4-IurR2d6-&si=iBYgrhtJiGNHQZbt'
    id = youtube.parse_youtube_url(playlist_url)
    if id[1]:
        videos = youtube.fetch_playlist_videos(id[0])
        playlist_name = youtube.get_playlist_name(id[0])
        print(f'Fetch playlist:  {playlist_name}/ {len(videos)}')
        for i, video in enumerate(tqdm(videos)):
            i_id = video['video_id']
            i_title = video['title']
            print(f'{i}. {i_title}')
            video_info = youtube.fetch_content(i_id)
            transcript = youtube.get_transcript(i_id)
            video_info['playlist'] = playlist_name
            video_info['summary'] = summarizer.summarize(transcript, i_title)
            log_youtube.save_to_notion_youtube(video_info)
            #'published_date': video.publish_date.isoformat() if video.publish_date else None,

def summarize_raindrop(config, summarizer, log_raindrop):
    raindrop = RaindropClient(config)
    raindrop_items = raindrop.fetch_content(None)
    # df_raindrop = pd.DataFrame(raindrop_items)
    texts = WebContent(config)
    for i,item in enumerate(raindrop_items):
        i_text = texts.fetch_content(item['url'])

        if len(i_text) > 300:
            raindrop_items[item]['summary']  = summarizer.summarize_content(i_text, item['title'])
        else:
            print('Too small to summarize.')
            raindrop_items[item]['summary'] = {'full_summary':'',
                               'one_sentence_summary':'',
                               'chapter_summary':''}
            
def summarize_web_text(processed_items, summarizer, extractor, logger, tags):
    for item in tqdm(processed_items, desc="Processing items"):
        try:
            # 2-1. 웹 콘텐츠 수집
            print(f"4. 웹 콘텐츠 수집 시작: {item['title']} - {item['url']}")
            
            article_url = item['url']
            # Extract the main text and essential details
            article_data = extractor.extract_text(article_url)
            article_data['text'] = extractor.clean_text(article_data['text'])
            
            # Display the extracted information
            if article_data:
                print(len(article_data['text']))
                if len(article_data['text']) > 10000:
                    print('Too long to summarize.')
                    continue
                print("Title:", article_data['title'])
                print("Author:", article_data['author'])
                print("Date:", article_data['date'])
                print("Text:", article_data['text'][:500], "...") 
            
            #content = pocket._fetch_single_content(item)
            
                item['content'] = article_data['text']
                item['title'] = article_data['title']
                item['author'] = article_data['author']
                item['date'] = article_data['date']
                # 2-2. 여기서 summarize 수행
                #utils.preprocess_text(item['content'])
                item['summary'] = summarizer.summarize(item['content'], item['title'])
                
                
                logger.save_to_notion_pocket(item)
            else:
                print(f"콘텐츠 수집 실패: {item['url']}")
                
        except Exception as e:
            print(f"아이템 처리 중 오류 발생 ({item['url']}): {str(e)}")
            continue
    #summarize_text(config, df_raindrop, texts, summarizer, log_raindrop, config.NOTION_DB_RAINDROP_ID)
def main():
    config = Config()
    summarizer = BaseSummarizer(config)
    log_youtube = YouTube2Notion(config)
    
    #summarize_youtube(config, summarizer, log_youtube)
#log_raindrop = Raindrop2Notion(config)
    pocket = PocketClient(config)
    extractor = WebContent(config)
    print("1. 설정 및 인스턴스 초기화 완료")
# 2. Pocket2Notion 인스턴스 생성 및 초기화

    logger = Pocket2Notion(config, verbose=True)
    logger.initialize(pocket)
    logger.change_id(config.NOTION_DB_POCKET_ID)
    
    print("2. Pocket2Notion 초기화 완료")
    #pocket_items = pocket.fetch_content(tags='_untagged_')
    # 여러 태그가 있는 아이템 검색
    tags = ['사회']
    print(f"3. 처리할 태그: {tags}")
    processed_items = pocket.fetch_content(tags=tags)
    summarize_web_text(processed_items, summarizer, extractor, logger, tags)
    #
    # 2. 각 아이템별 순차 처리
    

        


if __name__ == "__main__":
    main()
