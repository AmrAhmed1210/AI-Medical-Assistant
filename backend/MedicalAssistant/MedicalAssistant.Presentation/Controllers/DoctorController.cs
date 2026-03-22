using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers
{
    [ApiController]
    [Route("doctors")]   
    public class DoctorsController : ControllerBase
    {
        private readonly IDoctorService _doctorService;

        public DoctorsController(IDoctorService doctorService)
        {
            _doctorService = doctorService;
        }

        // GET /doctors
        // GET /doctors?specialtyId=1
        [HttpGet]
        public async Task<ActionResult<IEnumerable<DoctorDTO>>> GetAllDoctors(
            [FromQuery] int? specialtyId = null)
        {
            var doctors = specialtyId.HasValue
                ? await _doctorService.GetDoctorsBySpecialtyAsync(specialtyId.Value)
                : await _doctorService.GetAllDoctorsAsync();

            return Ok(doctors);
        }

        // GET /doctors/{id}
        [HttpGet("{id}")]
        public async Task<ActionResult<DoctorDetailsDTO>> GetDoctorById(int id)
        {
            var doctor = await _doctorService.GetDoctorByIdAsync(id);
            if (doctor == null)
                return NotFound(new { message = "Doctor not found." });
            return Ok(doctor);
        }
    }
}
