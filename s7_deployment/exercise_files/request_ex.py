import requests


def test_response():
    # response = requests.get('https://api.github.com/this-api-should-not-exist')
    # response = requests.get('https://api.github.com')
    # response=requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops")
    response = requests.get(
        'https://api.github.com/search/repositories',
        params={'q': 'requests+language:python'},
    )
        
    status_code = response.status_code
    
    if response.status_code == 200:
        print('Success!')
        
        print(f"\nContent\n{'-'*10}\n{response.json()}\n")
    elif response.status_code == 404:
        print('Not Found.')
        
def download_image():
    response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')    
    
    with open(r'img.png','wb') as f:
        f.write(response.content)

def test_post():
    pload = {'username':'Olivia','password':'123'}
    response = requests.post('https://httpbin.org/post', data = pload)
    
    print(f"Response: {response.content}")

if __name__ == "__main__":
    # test_response()
    # download_image()
    test_post()