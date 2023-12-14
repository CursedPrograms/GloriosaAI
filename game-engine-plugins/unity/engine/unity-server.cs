using UnityEngine;
using UnityEngine.Networking;

public class YourScript : MonoBehaviour
{
    void Start()
    {
        StartCoroutine(GetRequest("http://your-flask-server-address:5000"));
    }

    IEnumerator GetRequest(string uri)
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get(uri))
        {
            yield return webRequest.SendWebRequest();

            if (webRequest.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("Received: " + webRequest.downloadHandler.text);
            }
            else
            {
                Debug.Log("Error: " + webRequest.error);
            }
        }
    }
}
