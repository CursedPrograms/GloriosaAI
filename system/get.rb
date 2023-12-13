require 'net/http'

flask_url = 'http://localhost:5000/api/data'
uri = URI(flask_url)
response = Net::HTTP.get_response(uri)

if response.is_a?(Net::HTTPSuccess)
  puts "Response from Flask: #{response.body}"
else
  puts "Error: #{response.code} - #{response.message}"
end