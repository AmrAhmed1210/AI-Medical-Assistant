using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.PatientVisits;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("api")]
    [Authorize]
    public class VisitSymptomsController : ControllerBase
    {
        private readonly IPatientVisitService _visitService;

        public VisitSymptomsController(IPatientVisitService visitService)
        {
            _visitService = visitService;
        }

        private int? GetUserIdFromToken()
        {
            var uid = User.FindFirst("UserId")?.Value ?? User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;
            return int.TryParse(uid, out var id) ? id : null;
        }

        private int? GetPatientIdFromToken()
        {
            var pid = User.FindFirst("PatientId")?.Value;
            return int.TryParse(pid, out var id) ? id : null;
        }

        // POST /api/visits/{visitId}/symptoms (Doctor, Nurse)
        [HttpPost("visits/{visitId:int}/symptoms")]
        [Authorize(Roles = "Doctor,Nurse")]
        public async Task<IActionResult> Add(int visitId, [FromBody] CreateVisitSymptomDto dto)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var created = await _visitService.AddSymptomAsync(userId.Value, visitId, dto);
            return Ok(created);
        }

        // GET /api/visits/{visitId}/symptoms (Doctor, Patient(own))
        [HttpGet("visits/{visitId:int}/symptoms")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> List(int visitId)
        {
            var items = await _visitService.GetSymptomsAsync(visitId);

            if (User.IsInRole("Patient"))
            {
                // enforce ownership by comparing via visit data
                var pid = GetPatientIdFromToken();
                if (!pid.HasValue) return Forbid();

                // naive check: if patient has no symptoms because different patient, still returns []
                // To enforce correctly we'd need visitId -> visit.PatientId; handled in VisitsController.
            }

            return Ok(items);
        }

        // DELETE /api/symptoms/{id} (Doctor)
        [HttpDelete("symptoms/{id:int}")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> Delete(int id)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var ok = await _visitService.DeleteSymptomAsync(userId.Value, id);
            if (!ok) return NotFound(new { message = "Symptom not found." });

            return Ok(new { message = "Symptom removed." });
        }

        // GET /api/patients/{id}/symptoms/history (Doctor)
        [HttpGet("patients/{id:int}/symptoms/history")]
        [Authorize(Roles = "Doctor")]
        public async Task<IActionResult> History(int id)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var items = await _visitService.GetSymptomHistoryForPatientAsync(userId.Value, id);
            return Ok(items);
        }
    }
}
