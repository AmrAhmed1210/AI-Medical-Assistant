using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.DoctorDTOs;
using Microsoft.AspNetCore.Mvc;

namespace MedicalAssistant.Presentation.Controllers;

[ApiController]
[Route("api/specialties")]
public class SpecialtiesController : ControllerBase
{
    private readonly IDoctorService _doctorService;

    public SpecialtiesController(IDoctorService doctorService)
    {
        _doctorService = doctorService;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<SpecialtyDto>>> GetAll()
    {
        var specialties = await _doctorService.GetSpecialtiesAsync();
        return Ok(specialties);
    }
}
