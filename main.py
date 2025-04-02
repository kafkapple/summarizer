import os
from tqdm import tqdm
import json

from src.fetcher.base import WebContent
from src.fetcher.youtube import YouTube
from src.fetcher.raindrop import RaindropClient
from src.fetcher.pocket import PocketClient

#from src.summarizer import BaseSummarizer
from src.lang_sum import LangChainSummarizerSync as BaseSummarizer
from src.logger import Pocket2Notion, Raindrop2Notion, YouTube2Notion
from src.utils import Utils, setup_logging
import logging
import hydra
from omegaconf import DictConfig

# 로거 설정 (setup_logging 사용)
logger, debug_logger = setup_logging()
utils = Utils() # Utils 인스턴스 생성

# In main.py, replace this section in summarize_youtube function:
@utils.timeit
def summarize_youtube(cfg, summarizer, log_youtube, playlist_url):
    print("====================")
    youtube = YouTube(cfg)
    id, is_playlist = youtube.parse_youtube_url(playlist_url)
    
    if is_playlist:
        videos = youtube.fetch_playlist_videos(id)
        playlist_name = youtube.get_playlist_name(id)
        print(f'Fetch playlist: {playlist_name} / {len(videos)}')
        
        for i, video in enumerate(tqdm(videos)):
            i_id = video['video_id']
            i_title = video['title']
            print(f'{i}. {i_title}')
            
            # Fetch video info and transcript
            video_info = youtube.fetch_content(i_id)
            transcript = youtube.get_transcript(i_id)
            
            # Enhanced transcript validation with better debugging
            if transcript is None:
                print(f"Skipping {i_title} - No transcript available")
                continue
                
            # Make sure we have a string
            if not isinstance(transcript, str):
                print(f"Converting transcript type {type(transcript)} to string for {i_title}")
                transcript = str(transcript)
            
            # Check if transcript is empty after stripping whitespace
            if not transcript.strip():
                print(f"Skipping {i_title} - Empty transcript after cleaning")
                continue
                
            # Debug logging
            print(f"\nTranscript length: {len(transcript)}")
            print(f"Transcript sample: {transcript[:100]}...")
            
            # Validate video info
            if not video_info or not video_info.get('url'):
                print(f"Skipping {i_title} due to missing or empty URL.")
                continue
            if not video_info.get('title'):
                print(f"Skipping {i_title} due to missing or empty Title.")
                continue
            
            # Summarization with better error handling
            try:
                print(f"\nSummarizing {i_title}...")
                summary_result = summarizer.summarize(transcript, i_title)
                
                # 디버깅: 최종 요약 결과 로깅 (원본 텍스트 제외)
                if summary_result:
                    summary_to_log = summary_result.copy()
                    summary_to_log.pop('full_text', None)
                    logger.debug("--- Final Summary Result (YouTube) ---")
                    logger.debug(json.dumps(summary_to_log, indent=2, ensure_ascii=False))
                    print("--- Final Summary Result (YouTube) ---")
                    print(json.dumps(summary_to_log, indent=2, ensure_ascii=False))
                    print("----------------------------------------")
                else:
                    logger.warning("Summarization returned None result.")

                if summary_result is None:
                    print(f"Skipping {i_title} - Summarization failed")
                    continue
                    
                # Add additional info and save to Notion
                video_info['playlist'] = playlist_name
                # Add summary result to video_info for saving
                video_info['summary_result'] = summary_result 
                log_youtube.save_to_notion_youtube(video_info)
                
                # Save summary to files using Utils
                if summary_result:
                     # Add metadata to summary_result before saving
                    summary_result['metadata'] = {
                        'title': video_info.get('title', i_title),
                        'url': video_info.get('url', f'https://www.youtube.com/watch?v={i_id}'),
                        'channel': video_info.get('channel'),
                        'published_date': video_info.get('published_date'),
                        'duration': video_info.get('duration'),
                        'view_count': video_info.get('view_count'),
                        'like_count': video_info.get('like_count'),
                        'comment_count': video_info.get('comment_count'),
                        'category': video_info.get('category'),
                        'tags': video_info.get('tags', [])
                    }
                    summary_result['full_text'] = transcript # Include original text
                    utils.save_summary_to_files(summary_result)
                
                # 디버깅: 최종 요약 결과 로깅 (원본 텍스트 제외)
                if summary_result:
                    summary_to_log = summary_result.copy()
                    summary_to_log.pop('full_text', None)
                    logger.debug("--- Final Summary Result (Web) ---")
                    logger.debug(json.dumps(summary_to_log, indent=2, ensure_ascii=False))
                    print("--- Final Summary Result (Web) ---")
                    print(json.dumps(summary_to_log, indent=2, ensure_ascii=False))
                    print("------------------------------------")
                else:
                    logger.warning("Summarization returned None result.")
                
            except Exception as e:
                print(f"Error summarizing {i_title}: {str(e)}")
                continue

def summarize_web_text(processed_items, summarizer, extractor, logger, tags, cfg):
    for item in tqdm(processed_items, desc="Processing items"):
        try:
            print(f"4. 웹 콘텐츠 수집 시작: {item.get('Title', {}).get('title', [{}])[0].get('text', {}).get('content', 'N/A')} - {item.get('URL', {}).get('url', 'N/A')}")
            
            article_url = item.get('URL', {}).get('url')
            if not article_url:
                print(f"Skipping item due to missing URL.")
                continue

            article_data = extractor.extract_text(article_url)
            if not article_data:
                print(f"콘텐츠 수집 실패: {article_url}")
                continue
                
            article_data['text'] = extractor.clean_text(article_data['text'])
            
            if article_data['text']:
                print(len(article_data['text']))
                if len(article_data['text']) > 10000:
                    print('Too long to summarize.')
                    continue
                print("Title:", article_data['title'])
                print("Author:", article_data['author'])
                print("Date:", article_data['date'])
                print("Text:", article_data['text'][:500], "...") 
            
                item['content'] = article_data['text']
                # Ensure title is a simple string for summarizer
                item_title = article_data.get('title', item.get('Title', {}).get('title', [{}])[0].get('text', {}).get('content', ''))
                
                # 요약 시도 및 결과 검사
                print(f"\n--- Summarizing Web Content ---")
                print(f"Title: {item_title}")
                print(f"Content length: {len(item['content']) if item.get('content') else 0}")
                print(f"Content sample: {item['content'][:200] if item.get('content') else 'N/A'}...")
                summary_result = summarizer.summarize(item['content'], item_title)
                print(f"Summarization result type: {type(summary_result)}")
                print(f"Summarization result (first 100 chars): {str(summary_result)[:100] if summary_result else 'None'}...")
                print(f"------------------------------")
                
                # 디버깅: 최종 요약 결과 로깅 (원본 텍스트 제외)
                if summary_result:
                    summary_to_log = summary_result.copy()
                    summary_to_log.pop('full_text', None)
                    logger.debug("--- Final Summary Result (Web) ---")
                    logger.debug(json.dumps(summary_to_log, indent=2, ensure_ascii=False))
                    print("--- Final Summary Result (Web) ---")
                    print(json.dumps(summary_to_log, indent=2, ensure_ascii=False))
                    print("------------------------------------")
                else:
                    logger.warning("Summarization returned None result.")

                if summary_result is None:
                    print(f"Skipping {item_title} due to failed summarization (summary is None).")
                    continue
                
                # Update item with extracted data before saving
                item['Title']['title'][0]['text']['content'] = article_data.get('title', item_title)
                
                # Add summary result to item
                item['summary_result'] = summary_result 
                
                # Check Title and URL again before saving
                if not item.get('Title', {}).get('title', [{}])[0].get('text', {}).get('content'):
                     print(f"Skipping item due to missing or empty Title content before saving.")
                     continue
                if not item.get('URL', {}).get('url'):
                    print(f"Skipping item due to missing or empty URL before saving.")
                    continue
                    
                if cfg.source == 'pocket':
                    logger.save_to_notion_pocket(item)
                elif cfg.source == 'raindrop':
                     logger.save_to_notion_raindrop(item)
                     
                # Save summary to files using Utils
                if summary_result:
                     # Add metadata to summary_result before saving
                    summary_result['metadata'] = {
                        'title': item.get('Title', {}).get('title', [{}])[0].get('text', {}).get('content', 'N/A'),
                        'url': item.get('URL', {}).get('url', 'N/A'),
                        # Add other relevant metadata from 'item' if available
                    }
                    summary_result['full_text'] = item.get('content', '') # Include original text
                    utils.save_summary_to_files(summary_result)
            else:
                print(f"콘텐츠 수집 실패 또는 텍스트 없음: {article_url}")
                
        except Exception as e:
            print(f"아이템 처리 중 오류 발생 ({item.get('URL', {}).get('url', 'N/A')}): {str(e)}")
            continue
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # BASE_PATH 환경 변수 설정
    if 'BASE_PATH' not in os.environ:
        os.environ['BASE_PATH'] = os.path.dirname(os.path.abspath(__file__))
    
    # 필요한 디렉토리 생성
    if cfg.paths and cfg.paths.get('src'):
        os.makedirs(cfg.paths['src'], exist_ok=True)
 
    
    # 설정 로그 출력
    print("\n=== Configuration ===")
    print(f"Source: {cfg.source}")
    print(f"Keywords Enabled: {cfg.include_keywords}")
    print(f"Full Text Enabled: {cfg.include_full_text}")
    print(f"Chapters Enabled: {cfg.enable_chapters}")
    print(f"Base Path: {cfg.paths['base']}")
    print(f"Model: {cfg.llm['model']}")
    print("==================\n")
    
    summarizer = BaseSummarizer(cfg)
    
    if cfg.source == 'youtube':

        if not cfg.playlist_url:
            raise ValueError("YouTube 소스 선택 시 playlist_url 필수")
        log_youtube = YouTube2Notion(cfg)
        summarize_youtube(cfg, summarizer, log_youtube, cfg.playlist_url)
    
    elif cfg.source == 'pocket':
        pocket = PocketClient(cfg)
        extractor = WebContent(cfg)
        logger = Pocket2Notion(cfg, verbose=cfg.verbose)
        logger.initialize(pocket)
        logger.change_id(cfg.notion.pocket_id)
        
        processed_items = pocket.fetch_content(tags=cfg.tags)
        summarize_web_text(processed_items, summarizer, extractor, logger, cfg.tags, cfg)
        
    elif cfg.source == 'raindrop':
        raindrop = RaindropClient(cfg)
        extractor = WebContent(cfg)
        logger = Raindrop2Notion(cfg, verbose=cfg.verbose)
        logger.change_id(cfg.notion.raindrop_id)
        
        processed_items = raindrop.fetch_content()
        summarize_web_text(processed_items, summarizer, extractor, logger, cfg.tags, cfg)


if __name__ == "__main__":
    main()
