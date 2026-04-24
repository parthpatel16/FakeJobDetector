const fs = require('fs');
const indexHtml = fs.readFileSync('vercel_index.html', 'utf8');
const landingHtml = fs.readFileSync('vercel_landing.html', 'utf8');

const getStyle = html => html.match(/<style>([\s\S]*?)<\/style>/)[1];
const css = getStyle(indexHtml) + '\n' + getStyle(landingHtml);
fs.writeFileSync('index.css', css.replace(/<!--[\s\S]*?-->/g, ''));

const getBody = html => html.match(/<body>([\s\S]*?)<script>/)[1]
  .replace(/class=/g, 'className=')
  .replace(/style=\"([^\"]*?)\"/g, (m, styles) => {
    const obj = {};
    styles.split(';').forEach(s => {
      const [k, v] = s.split(':');
      if (k && v) obj[k.trim().replace(/-([a-z])/g, g => g[1].toUpperCase())] = v.trim();
    });
    return 'style={{' + JSON.stringify(obj).slice(1,-1).replace(/"([^"]+)":"([^"]+)"/g, '$1:"$2"') + '}}';
  })
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/<img(.*?)>/g, '<img$1 />')
  .replace(/<input(.*?)>/g, '<input$1 />')
  .replace(/<br>/g, '<br />')
  .replace(/<hr(.*?)>/g, '<hr$1 />')
  .replace(/<path(.*?)>/g, '<path$1 />')
  .replace(/<rect(.*?)>/g, '<rect$1 />')
  .replace(/<circle(.*?)>/g, '<circle$1 />')
  .replace(/<line(.*?)>/g, '<line$1 />')
  .replace(/<defs(.*?)>([\s\S]*?)<\/defs>/g, '<defs$1>$2</defs>')
  .replace(/<stop(.*?)>/g, '<stop$1 />');

fs.writeFileSync('index_jsx.txt', getBody(indexHtml));
fs.writeFileSync('landing_jsx.txt', getBody(landingHtml));
console.log('Done!');
