using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/test")]
[Authorize] // MUST be authorized to access JWT claims
public class TestController : ControllerBase
{
    [HttpGet("patient-summary")]
    public IActionResult GetPatientSummary()
    {
        // 🔒 SECURITY RULE APPLIED:
        // PatientId is extracted strictly from the JWT Claims.
        // It is PROHIBITED to accept [FromQuery] int patientId from the client.
        var claim = User.FindFirst("PatientId")?.Value;
        
        if (!int.TryParse(claim, out var patientId) || patientId <= 0)
        {
            return Unauthorized(new { error = "Unauthorized access or missing PatientId in token." });
        }

        // Mock data response based on the securely extracted patientId
        var mockData = new
        {
            Message = "This is a secure mock response.",
            PatientId = patientId, 
            Age = 35,
            Name = "Fake Patient Name" // As requested for the mock
        };

        return Ok(mockData);
    }
}
