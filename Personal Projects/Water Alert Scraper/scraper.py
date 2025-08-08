import requests
from bs4 import BeautifulSoup
import time, sys, os

container_path = './container'
last_container_path = './last_container'
keywords = ['rotura', 'interrupção', 'reparação', 'abastecimento', 'corte', 'suspensão', 'manutenção', 'avaria', 'emergência']
interval_seconds = 6 * 3600  # 6 hours
timestamp_file   = './last_run_timestamp.txt'
webhook_url = 'DISCORD_WEBHOOK_URL_HERE'

def already_ran():
    """Return True if script ran less than interval_seconds ago."""
    try:
        last = float(open(timestamp_file).read())
    except Exception:
        return False
    return (time.time() - last) < interval_seconds

def mark_ran():
    with open(timestamp_file, 'w') as f:
        f.write(str(time.time()))


def save_html(html, path):
    with open(path, 'wb') as f:
        f.write(html)

def open_html(path):
    if not os.path.exists(path):
    	return b''
    with open(path, 'rb') as f:
        return f.read()
    
def is_different():
    current = open_html(container_path)
    last = open_html(last_container_path)
    return current != last

def has_keyword(text):
    text = str.lower(text)
    return any(keyword in text for keyword in keywords)
    
def truncate_text(text):
    # Remove leading/trailing whitespace and split into lines
    lines = text.strip().splitlines()
    # Remove empty lines
    non_empty_lines = [line for line in lines if line.strip()]
    # Find start and end indices
    start_index = 0  #starts from the beginning (date)
    end_index = None
    for i, line in enumerate(non_empty_lines):
        if 'Compartilhar' in line:
            end_index = i   #Exclude the "Compartilhar" line
            break

    #newline and join the lines to get the final result
    if end_index is not None:
        result = '\n'.join(non_empty_lines[start_index:end_index])
    else:
        result = '\n'.join(non_empty_lines[start_index:])

    return result
    
def make_request(url):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
    r = requests.get(url, headers=headers, verify=False)
    if (r.status_code != 200):
        print("ERROR! Code " + r.status_code)
    return r

def send(text):
    if not webhook_url.startswith("http"):
        print("Add your discord webhook to the code first!")
        return
    try:
        response = requests.post(webhook_url, json={"content": f"🚰 Water Alert:\n```{text}```"})
        if response.status_code >= 400:
            print(f"Failed to send alert. Status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Exception while sending alert: {e}")


def main():
    if already_ran():
        sys.exit(0)
    mark_ran()
    url = 'https://www.smasalmada.pt/avisos'
    r = make_request(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    text = soup.select_one('#aviso_right_0 .container').getText()
    save_html(text.encode('utf-8'), container_path)
    if (is_different()):
        save_html(text.encode('utf-8'), last_container_path)
        if (has_keyword(text)):
            text = truncate_text(text)
            send(text)

main()
