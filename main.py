from tqdm import tqdm 
import os
import pandas as pd 
from config import Config
from fetch import YouTube, WebContent, RaindropClient, PocketClient#, DiffbotExtractor
from summarizer import BaseSummarizer#, TextSummarizer, VideoSummarizer
from logger import Pocket2Notion, YouTube2Notion, Raindrop2Notion
from utils import Utils
import logging
import argparse
logging.basicConfig(level=logging.INFO)
utils = Utils()

def parse_arguments():
    parser = argparse.ArgumentParser(description='콘텐츠 요약 도구')
    
    # 소스 관련 인자
    parser.add_argument('--source', type=str, required=True,
                        choices=['youtube', 'pocket', 'raindrop'],
                        help='요약할 콘텐츠 소스 선택')
    parser.add_argument('--playlist_url', type=str,
                        help='YouTube 재생목록 URL')
    parser.add_argument('--tags', nargs='+', default=['_untagged_'],
                        help='Pocket/Raindrop 태그 필터')

    # 요약 옵션
    parser.add_argument('--keywords', action='store_true', default=True,
                        help='키워드 추출 포함')
    parser.add_argument('--no-keywords', action='store_false', dest='keywords',
                        help='키워드 추출 비활성화')
    
    parser.add_argument('--full-text', action='store_true', default=False,
                        help='전체 텍스트 포함')
    parser.add_argument('--no-full-text', action='store_false', dest='full_text',
                        help='전체 텍스트 제외')
    
    parser.add_argument('--chapters', action='store_true', default=True,
                        help='챕터별 요약 활성화')
    parser.add_argument('--no-chapters', action='store_false', dest='chapters',
                        help='챕터별 요약 비활성화')
    
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='상세 로그 출력')
    
    return parser.parse_args()

@utils.timeit
def summarize_youtube(config, summarizer, log_youtube, playlist_url):
    youtube = YouTube(config)
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
    args = parse_arguments()
    config = Config()
    
    # Config 객체에 실행 시 설정 적용
    config.update_runtime_settings(
        keywords=args.keywords,
        full_text=args.full_text,
        chapters=args.chapters
    )
    
    summarizer = BaseSummarizer(config)
    
    if args.source == 'youtube':
        if not args.playlist_url:
            raise ValueError("YouTube 소스 선택 시 --playlist_url 필수")
        log_youtube = YouTube2Notion(config)
        summarize_youtube(config, summarizer, log_youtube, args.playlist_url)
    
    elif args.source == 'pocket':
        pocket = PocketClient(config)
        extractor = WebContent(config)
        logger = Pocket2Notion(config, verbose=args.verbose)
        logger.initialize(pocket)
        logger.change_id(config.NOTION_DB_POCKET_ID)
        
        processed_items = pocket.fetch_content(tags=args.tags)
        summarize_web_text(processed_items, summarizer, extractor, logger, args.tags)

if __name__ == "__main__":
    main()
