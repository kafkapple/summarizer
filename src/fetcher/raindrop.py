from src.fetcher.base import WebContent
from omegaconf import DictConfig
from src.utils import Utils
from tqdm import tqdm
import requests
from typing import List, Dict

class RaindropClient(WebContent):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.token = cfg.api_keys['raindrop']
        self.base_url = "https://api.raindrop.io/rest/v1/"
        self.filename_collection = 'raindrop_collections.csv'
        self.filename_item = 'raindrop_items.csv'

    def fetch_content(self, identifier):
        """Raindrop API를 통해 아이템 가져오기"""
        items = self._fetch_raindrop_items(identifier)
        
        # 각 아이템의 웹 컨텐츠 가져오기
        for item in items:
            try:
                item['content'] = self.fetch_web_content(item['link'])
            except Exception as e:
                print(f"Error fetching content for {item['link']}: {e}")
                item['content'] = None
                
        return items

    def _fetch_raindrop_items(self, identifier):
        """Raindrop API에서 아이템 가져오기"""
        url = f"{self.base_url}raindrops/{identifier}"
        response = requests.get(url, headers={"Authorization": f"Bearer {self.token}"})
        response.raise_for_status()
        return response.json().get('items', [])

    def scrape_save(self):
        ## Export result
        bookmarks = self.get_bookmarks()
        collections = self.get_collections()
        items = self.get_item_from_collection(collections)
        try:
            Utils.save_file(collections, self.filename_collection)
            Utils.save_file(items, self.filename_item)
        except:
            print('save error')
 
    def get_collections(self):
        url = self.base_url +"collections"
        collections = self.get_response(url)

        print('Collections: ', len(collections))
        results = []
        for collect in collections:
            dict_collect = {
                'title': collect['title'],
                'id': collect['_id'],
                'count': collect['count'],
                'expanded': collect['expanded'],
                #'access': collect['access'],
                'parent':''
                }

            print(dict_collect)
            results.append(dict_collect)

        collections_child = self.get_child_collections()

        results+=collections_child
        return results

    def get_child_collections(self):
        url = self.base_url +"collections/childrens"
        collections = self.get_response(url)
        print('Collections: ', len(collections))
        results = []
        for collect in collections:
            dict_collect = {
                'title': collect['title'],
                'id': collect['_id'],
                'count': collect['count'],
                'expanded': collect['expanded'],
               # 'access': collect['access'],
                'parent':collect['parent']
                }
            try: 
                dict_collect['parent'] =collect['parent']['$ref']+ '_'+str(collect['parent']['$id'])
            except:
                print('parent parsing error.')
            print(dict_collect)
            results.append(dict_collect)
        return results
        
    def get_item_from_collection(self, collection_id: str) -> List[Dict]:
        """컬렉션의 아이템을 Notion 데이터베이스 속성에 맞게 처리"""
        items = self.get_collection_items(collection_id)
        processed_items = []
        
        for item in items:
            processed_item = {
                'Title': {'title': [{'text': {'content': item.get('title', '')}}]},
                'URL': {'url': item.get('link', '')},
                'Description': {'rich_text': [{'text': {'content': item.get('excerpt', '')}}]},
                'Category': {'multi_select': [{'name': item.get('collection', {}).get('title', '')}]},
                'Favorite': {'checkbox': item.get('favorite', False)},
                'Status': {'select': {'name': 'unread'}},
                'Word Count': {'number': item.get('word_count', 0)},
                'Tags': {'multi_select': [{'name': tag} for tag in item.get('tags', [])]},
                'Date': {'date': {'start': item.get('created', '')}},
                'Author': {'rich_text': [{'text': {'content': item.get('author', '')}}]},
                'Source': {'select': {'name': 'raindrop'}}
            }
            processed_items.append(processed_item)
            
        return processed_items
    
