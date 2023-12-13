using System;
using System.Diagnostics;

class Program
{
    static void Main()
    {
        Console.Clear();
        Console.WriteLine("=== My C# Console App ===");
        Console.WriteLine("1. Run Python Script");
        Console.WriteLine("2. Option 2");
        Console.WriteLine("3. Option 3");
        Console.WriteLine("0. Exit");

        Console.Write("Select an option: ");
        string userInput = Console.ReadLine();

        switch (userInput)
        {
            case "1":
                RunPythonScript();
                break;

            case "2":
                Console.WriteLine("You selected Option 2.");
                break;

            case "3":
                Console.WriteLine("You selected Option 3.");
                break;

            case "0":
                Console.WriteLine("Exiting the application. Goodbye!");
                return;

            default:
                Console.WriteLine("Invalid option. Please try again.");
                break;
        }

        Console.WriteLine("\nPress Enter to continue...");
        Console.ReadLine();
        Main();
    }

    static void RunPythonScript()
    {
        string pythonScript = "your_python_script.py";

        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = "python",
            Arguments = pythonScript,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using (Process process = new Process { StartInfo = psi })
        {
            process.Start();
            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            Console.WriteLine("Output from Python script:\n" + output);
        }
    }
}