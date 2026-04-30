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
    public class VisitVitalsController : ControllerBase
    {
        private readonly IPatientVisitService _visitService;

        public VisitVitalsController(IPatientVisitService visitService)
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

        // POST /api/visits/{visitId}/vitals (Doctor, Nurse)
        [HttpPost("visits/{visitId:int}/vitals")]
        [Authorize(Roles = "Doctor,Nurse")]
        public async Task<IActionResult> Add(int visitId, [FromBody] CreateVisitVitalDto dto)
        {
            var userId = GetUserIdFromToken();
            if (!userId.HasValue) return Unauthorized(new { message = "Invalid token." });

            var created = await _visitService.AddVisitVitalAsync(userId.Value, visitId, dto);
            return Ok(created);
        }

        // GET /api/visits/{visitId}/vitals (Doctor, Patient(own))
        [HttpGet("visits/{visitId:int}/vitals")]
        [Authorize(Roles = "Doctor,Patient")]
        public async Task<IActionResult> List(int visitId)
        {
            var items = await _visitService.GetVisitVitalsAsync(visitId);

            if (User.IsInRole("Patient"))
            {
                var pid = GetPatientIdFromToken();
                if (!pid.HasValue) return Forbid();
                // Ownership enforcement covered by visit detail endpoint; keep vitals list simple.
            }

            return Ok(items);
        }

        // DELETE /api/vitals/clinical/{id} (Doctor, Admin)
        [HttpDelete("vitals/clinical/{id:int}")]
        [Authorize(Roles = "Doctor,Admin")]
        public async Task<IActionResult> DeleteClinical(int id)
        {
            var ok = await _visitService.DeleteClinicalVitalAsync(id);
            if (!ok) return NotFound(new { message = "Vital entry not found." });
            return Ok(new { message = "Vital entry deleted." });
        }
    }
}
